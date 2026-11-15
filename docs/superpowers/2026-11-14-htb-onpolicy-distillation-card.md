# Experiment card H-OP: distillation on fresh self-play from the current `hard` (revision 2)

**Status:** DRAFT for review, 2026-11-14. **Nothing is built, generated, labelled, trained, exported or played under
this card.** It follows the student–teacher benchmark (ledger entry 65, #195), whose arm N result, TRAILS, supports
proposing this experiment. That result authorizes nothing beyond the proposal.

**Approvals this card needs before any work:** the card itself, and the narrow #53 and #87 exceptions in §2. A build
would then return once for launch review. Every runtime step in §15 needs its own approval.

**Revision 2 (2026-11-14, the user's review), before any build: the generation smoke (§4.3, §4.4).** Unchanged: the
40-game smoke, G_MAX 4,000, the 40 h cap, the row targets, the imitation gate and the match rule.
- **Y3 and Y4 now estimate usable rows per game separately for training and validation.** The game count is
  projected with the actual 9:1 split, and the slower pool sets it.
- **E4 applies only to validation,** as in final selection. A validation key found in any training game, later games
  included, is removed.
- **Y5 uses elapsed time for the full generation loop** (play, counting and recording), not search time per ply.
- **The 60/20-ply example is pinned** as a model-free regression that must return REVIEW.

**Revision 1 (2026-11-14, the user's review), before any build.** Unchanged: the row counts, the GPU generation path,
the imitation thresholds and the 272-pair match.
- **Rationale corrected (§1).**
  - Arm N measured a playing-strength gap in raw-policy games. It did not measure agreement at positions from `hard`'s
    self-play; that agreement is an open question, which the fit's reference scores answer first.
  - GPU generation is described as self-play under the research implementation of `hard`'s protocol. G1 and
    V2-R/V3 checked parts of that; real self-play trajectories are unverified (§4.1).
- **Generation smoke checks usable data yield (§4.4, §12).**
  - The smoke grows to 40 games. It estimates the yield of distinct, non-excluded keys per searched ply, and from it
    the time and games needed to reach both pools' targets, under the actual exclusions and mirror deduplication.
  - A low or uncertain estimate sends the plan back for review. The 4,000-game and 40 h caps stay.
- **Match interpretation made precise (§10, §11, §14).**
  - 0.60 is the planning target. A PASS establishes superiority over 0.50 under this protocol, and no more.
  - The 80% power figure is model-based and assumes the match completes and clears its guards.
  - The match tests the whole update: the new data source and the continued training together.
- **Evaluation split and wording (§10, §14).**
  - G3's openings are added to the match-opening exclusions.
  - For consistency they are also added to the generation and smoke openings. G3's positions were already excluded
    from both pools as E2.
  - A gate failure is reported as the challenger missing the imitation gate, with each metric's movement.

## 1. Question, hypothesis and mechanism

**The specific change.** The first H-TB fit learned from positions in `calib020_0001`'s old self-play corpus. This
card labels positions from **self-play under the research implementation of `hard`'s protocol**, using the same
teacher, and continues training the current student on them. Everything else stays as close to the first fit as
practical (§7).

**Hypothesis H-OP.** A challenger, warm-started from the current student and fitted to the pinned teacher's
mirror-averaged policy and value at positions from that self-play, is **stronger than the current `hard`** (true mean
pair score above 0.50) in a fresh, colour-reversed match under the same 200-simulation protocol. The **planning
target** is a true mean pair score of **0.60** (§11). The test covers the whole update. The new data source and the
continued training are not separated, so no result attributes an effect to either one alone.

**Why it is plausible:**
- **Distribution shift.** An imitator trained on someone else's positions can be wrong where it actually plays, and
  those errors compound over a game. This is the standard argument for DAgger-style on-policy labelling. The first fit
  never trained on a position from `hard`'s self-play.
- **A playing-strength gap against the teacher remains.** In the benchmark's raw-policy games, the student's
  mirror-averaged policy argmax scored **0.045** against the raw teacher it imitates (arm N, entry 65). That is a
  measured gap in **play**.
  - It is consistent with disagreement at the positions the student reaches, but it **does not measure** that.
  - How well the student agrees with the teacher on `hard` self-play is **an open question**. It is answered first
    by the reference arm's own scores in the fit: the start check, and KL_S, TOP_S and MSE_S on the 12,000 validation
    rows (§8).
  - The only agreement measured so far is top-1 0.651 on the first fit's validation rows (entry 52). Those were
    `calib020` corpus positions.
- **Search did not make up the difference.** Arm P minus arm N was +0.070 on common openings (descriptive).
- **Imitating this teacher has transferred to play before.** The first fit's student beat the original proven
  combination by 0.989 (entry 54) and the previous `hard` by 0.932 (G3, entry 63).
- **If agreement on `hard`'s positions turns out already high,** the room for this update is small. The gate's bar
  (I1, a 10% KL cut) then decides whether a match is proposed at all.

**What could make it fail, and what each outcome cannot separate:**
- **Capacity.** The student is 128 channels × 6 blocks, the teacher 48 filters × 20 blocks with 5×5 kernels. The
  remaining gap may be a capacity limit that more data cannot move.
- **The teacher was trained with own-link crossings allowed** (benchmark card §3). Its labels may be mis-specified
  in a way more fitting cannot fix.
- **Diminishing returns.** Closer imitation may no longer buy playing strength at this level.
- **The challenger misses the imitation gate (§8):** this cannot separate the dose, capacity and label quality.
- **The gate passes and the match does not:** imitation improved without a gain the match detected. That does not
  show no gain. The planning power is about 80% at 0.60 and much less below it (§11).
- **A PASS** shows the updated network beats the current one under this protocol. It does not show that on-policy
  data, rather than continued training on teacher labels, caused the gain.

**This card is not:** a change to the teacher, its configuration or its role. The teacher plays no game here, so no
#52 exception is needed. It is also not a deployment step: PASS would only support proposing an integration decision
(§14).

## 2. Ledger items this card touches, and the exceptions it needs

- **#53 (twixtbot as a training source) and #87 (distillation).** Each still needs an explicit decision. The H-TB
  exceptions covered the first card and its build only (#167). **Requested: narrow #53 and #87 exceptions for this
  card, its build and the runs in §15, once each.** They do not cover a second round, another teacher configuration,
  another recipe or any rescue after a result.
- **#195 (the benchmark is spent).** Its games and openings are excluded from generation, training and evaluation
  (§4.2, §5.3). The challenger plays the current `hard`, not the teacher.
- **#194 (Diagnostic O).** Its 1,100 positions and every G3 game are excluded from training and evaluation (§5.3).
- **#184 and #188 (the validated student and the pinned pair).** The student file `0e1cb621…` and the hard model
  `78c16a31c3dba37f` stay unchanged. The challenger is a new file, warm-started from a copy.
- **#186 and #187 (the runtime canary).** Any Python ORT parity surface runs only after the two-conv canary passes;
  `onnxruntime-node` stays pinned at exactly 1.23.2.
- **#165 (the policy-improvement recipe).** Not revived: that recipe trained a policy head toward search targets.
  This card fits the whole network to the external teacher at new positions.

## 3. Identities (fixed before any work)

| Role | Identity |
|---|---|
| Current student (MLX) | `~/twixt-cache/htb-pilot/student.safetensors`, sha256 `0e1cb621…`, state digest `586dca4f…` |
| Current `hard` (product) | ONNX pair `78c16a31c3dba37f` (exported from that student), served by `protocolMove`: mirror averaging, 200 simulations, T 1.0 sampling through ply 19, root sequential halving from ply 20 |
| Teacher | the first fit's label path, unchanged: twixtbot-ui `83749f23…` `model/pb` (pins `1958f847…`, `e0c2b882…`, `1d4073c4…`), raw network, rotation off, no search, mirror-averaged (pilot card §3, §5) |
| Teacher runtime | `~/twixt-cache/htb-teacher/teacher-venv`: Python 3.13.15, TensorFlow 2.21.0, NumPy 2.5.2 (checked before load) |
| Research runtime | project `.venv`: Python 3.14.10, MLX 0.30.3, NumPy 2.4.1, `MLX_ENABLE_TF32=0` |
| Product runtime | Node as pinned for G3 and the benchmark, `onnxruntime-node` 1.23.2 |
| Challenger | created by §7, saved create-only to `~/twixt-cache/htb-onpolicy/challenger.safetensors`; its export in §9 |

## 4. Generation: self-play by the current `hard`

### 4.1 Player and path

- **Self-play under the research implementation of `hard`'s protocol.** Both sides are the current student (MLX,
  `0e1cb621…`) with mirror averaging, 200 simulations, T 1.0 sampling through ply 19, root sequential halving from
  ply 20, and the research tie rule.
- **Path (kept): the research MLX path on the GPU**, in one process, as the H-TB screen and confirmation test ran it.
  It uses `MirrorAveragingEvaluator` and `RootSequentialHalvingMCTS`, checks the student's digest at load and at the
  end, and sets `MLX_ENABLE_TF32=0`.
  - **Why:** it was measured at 0.536 s per searched ply (entry 54), against 1.346 s on the product's Node path
    (benchmark arm P).
  - **What has been verified, and what has not:**
    - **G1** showed move-for-move agreement between this protocol and the product's `protocolMove` **with fake
      evaluators** (seed-for-seed fixtures, ties included).
    - **V2-R and V3** showed close output parity between the MLX student and the ONNX pair **on sampled real
      positions** (about 2.4e-5; mirror-averaged top-1 100%).
    - **Neither verifies real self-play trajectories.** With real networks, a tiny numeric difference can flip a
      close visit count or a T 1.0 sample, and the game then diverges. So the generated games are self-play under
      the research implementation, not games the product would have played.
    - The data needs `hard`'s position distribution, not its exact games. Whether the two implementations' position
      distributions differ materially is not tested here. The **match** (§10) runs on the product path, so the test
      of the update does not rest on this assumption.
  - The build reruns G1's fixtures against the generation driver's move function, so the driver is checked against
    `protocolMove` at the same (fake-evaluator) level.
- **Rules and game end:** our rules (red first, no swap, `allow_scl=False`), and the product's end rule only: play
  until a connection, or a draw when no legal moves remain. No ply cap.

### 4.2 Openings, seeds and the split

- **Openings:** each game starts from a supplied 4-ply opening, by the benchmark's rule (benchmark card §6): cells in
  rows and columns 4–19, index `int(sha256("htb-onpolicy/20261114/gen opening <i>/ply <k>")[:12], 16) mod n`.
  - Skipped if it equals, directly or reflected, an earlier generation opening, **any of the benchmark's openings**
    (its 200 match openings and its smoke openings), or **any G3 opening**. A G3 opening is the first four plies of
    each of G3's 220 match games and its smoke pair. G3 started from the empty board, so these are recorded moves,
    not supplied openings.
  - Why openings at all: `hard`'s empty-board policy is peaked, and 28 of G3's 220 games were exact copies. Plies 0–3
    are never labelled.
- **Seeds:** `sha256("htb-onpolicy/20261114/gen/game <i>/ply <k>")`. Hex digits 0–11 are the search seed and 12–23
  the readout seed, as in G3 and the benchmark. The namespace is new; the build checks all reachable seeds are
  distinct and below 2^48. These are sha256 namespaces like G3's, not 2027-band blocks.
- **Split by whole game, fixed before play:** game *i* is a **validation** game if *i* mod 10 = 9, otherwise a
  **training** game. No game contributes to both pools.

### 4.3 How many games: a stopping rule on position counts, not outcomes

- Games are generated serially in index order, each one deterministic given its seeds.
- After each game, the driver counts distinct canonical keys (§5.1) at plies 4 to *n*−1, net of the exclusion sets
  as selection applies them (§5.3: E1–E3 for training; E1–E4 for validation):
  - in the training games;
  - in the validation games, leaving out any key found in **any** training game generated so far. A validation key
    that first appears in a later training game leaves the count at that point.
  - When generation stops there are no later games, so this count equals the final selection's validation pool.
- **Generation stops** after the first game at which both counts reach **1.15 × N** (138,000 and 13,800), or at
  **G_MAX = 4,000 games**, whichever comes first. The rule reads position counts only, never outcomes.
- **Planning figure:** about 3,030 games, at G3's 50.7 searched plies a game (plies 4 to *n*−1).
  - Game length is heavy-tailed. G3's median was 39 searched plies, and its four board-full draws supplied 20% of
    all positions. Benchmark arm P's median was 31, with board-full draws supplying 31%.
  - So the game count to reach the targets is uncertain, and the smoke (§4.4) estimates it before launch.
- **Reaching G_MAX is not a failure in itself.** A pool short of N is refused at selection (§5.2).

### 4.4 The generation smoke: usable rows per game, by pool; games and time (frozen; revision 2)

**Smoke games:**
- **40 games**, indices *j* = 0–39, in their own namespaces (`…/gen smoke opening <j>/…`, `…/gen smoke/game <j>/…`).
- Openings are excluded exactly as the run's are (§4.2), and also from the run's own generation openings.
- Arm labels follow the run's rule: *j* mod 10 = 9 is validation-like (4 games), the rest training-like (36 games).
- **The smoke runs the run's generation loop unchanged:** play, then count keys, then write the game's records.
- Smoke games are never used for training or evaluation. Their outcomes are not read. The analyser reads lengths,
  keys and wall times only.

**Usable rows, counted as final selection counts them (§5.2, §5.3):**
- **Training-like game *j*:** y_j is the number of keys at its plies 4 to *n*−1 that are:
  - not in E1–E3 (**E4 does not apply to training**, as in selection);
  - not already counted earlier in this game or in an earlier training-like smoke game (mirror canonical key,
    first occurrence in index order).
- **Validation-like game *j*:** y_j is the number of keys at its plies 4 to *n*−1 that are:
  - not in E1–E4;
  - **not present in any training-like smoke game, whatever its index, earlier or later**;
  - not already counted earlier in this game or in an earlier validation-like smoke game.

**Measured, by the frozen analyser (`yield` command) from the committed smoke evidence:**

| Quantity | Definition |
|---|---|
| x_j | searched plies in smoke game *j* (plies 4 to *n*−1) |
| u_tr, u_val | usable rows per game: the mean of y_j over the 36 training-like and the 4 validation-like games |
| f_tr, f_val | Σ y_j / Σ x_j within each pool: usable rows per searched ply |
| σ | the standard deviation of min(y_j, 150) over all 40 games, each game's y_j taken in its own pool |
| u_tr,LB, u_val,LB | the pool's mean of min(y_j, 150), minus 1.645 × σ / √n_pool (n_pool = 36 and 4) |
| t0 | startup: process start to the start of smoke game 0 (model load, digest checks, building E1–E4) |
| w | loop seconds per game: the start of game 0 to the end of game 39's records, divided by 40 |

- **Why the lower bounds have this form:**
  - Truncating at 150 can only lower a game's count, so the truncated mean bounds the true mean from below. It also
    keeps rare 572-ply draws from inflating σ.
  - **Each pool keeps its own mean,** so a pool that is slower in the smoke is not averaged away (the 60/20 example
    below).
  - The **spread is pooled.** Arms are assigned by index before play, so their game lengths share one distribution
    by construction, and four games alone give a very noisy spread.
  - 1.645 is a one-sided normal factor. This is a planning rule, not a guarantee.

**Projected game counts, with the actual 9:1 split.** Among the first G games there are
n_tr(G) = G − ⌊G/10⌋ training games and n_val(G) = ⌊G/10⌋ validation games. The **slower pool sets the count**:
- **Point, to the stop targets:** G_stop = max( min{G : n_tr(G) × u_tr ≥ 138,000}, min{G : n_val(G) × u_val ≥
  13,800} ).
- **Conservative, to the exact counts:** G_N = max( min{G : n_tr(G) × u_tr,LB ≥ 120,000}, min{G : n_val(G) ×
  u_val,LB ≥ 12,000} ). A pool whose lower bound is ≤ 0 makes G_N infinite.
- The record names the pool that sets each count.

**Time, from elapsed time for the full generation loop:** T = t0 + G_stop × w.
- w includes play, key counting against E1–E4 and the training keys, and writing records, because the smoke times
  the run's loop exactly.
- Counting at the run's scale (hash-set lookups over about 150,000 keys) is not measured by the smoke. The 1.5
  factor covers it.

**Verdict, frozen:** **LAUNCH_OK** needs every one of these. Otherwise **REVIEW**, which lists each failed condition
with its numbers and the pool that set it.

| # | Condition | Guards against |
|---|---|---|
| Y1 | all 40 smoke games complete, one record per searched ply, and Σ x_j ≥ 800 | an unusable smoke |
| Y2 | f_tr ≥ 0.90 and f_val ≥ 0.90 | low yield in either pool after exclusions and deduplication |
| Y3 | G_N ≤ 4,000 | the G_MAX cap leaving either pool short of N at a conservative yield |
| Y4 | G_stop ≤ 4,000 | either pool's stop target lying beyond G_MAX even at the point estimate |
| Y5 | 1.5 × T ≤ 144,000 s (40 h) | the fixed ceiling, on full-loop time |

- **REVIEW means no launch.** The smoke evidence and the yield record are committed, the plan returns to the user,
  and nothing is rescaled, retried or relaunched without a reviewed decision.
- **The caps stay:** G_MAX 4,000 and the fixed 40 h ceiling.

**Pinned regression: the 60/20-ply example returns REVIEW (model-free; part of the build).** The `yield` analyser
gets synthetic smoke evidence:
- 36 training-like games of 60 searched plies and 4 validation-like games of 20 searched plies;
- every key distinct and outside E1–E4;
- t0 = 10 s and w = 1 s.

The test asserts:
- u_tr = 60 and u_val = 20; f_tr = f_val = 1.
- G_stop = 6,900, set by validation: 690 validation games are needed, and training alone would need 2,555.
- G_N = 12,000, set by validation: u_val,LB = 20 − 1.645 × 12.153 / 2 = 10.004.
- **Verdict REVIEW, with exactly Y3 and Y4 failed,** both naming the validation pool. Y1, Y2 and Y5 hold, with
  T = 6,910 s.

Revision 1's pooled formula would have returned LAUNCH_OK here (G_stop 2,739, G_N 2,524). It averaged the slow
validation games into the fast training games.

The build adds two more model-free tests:
- a key in E4 still counts for training but not for validation;
- a validation key that first appears in a **later** training-like game is removed.

**Planning check, not a prediction.** I drew 4,000 random 40-game smokes with replacement from earlier match lengths
(searched plies = plies − 4, every key usable). Self-play length is unknown.

| Lengths drawn from | Y3 passes | Y4 passes | Y3 and Y4 | median G_N |
|---|---|---|---|---|
| G3's 220 games | 44% | 88% | 42% | 4,150 |
| Benchmark arm P's 220 games | 3% | 39% | 2.5% | 6,070 |

**With only 4 validation-like games, the validation lower bound usually decides Y3,** and REVIEW is the more likely
smoke outcome unless self-play games run longer than G3's.

## 5. Position selection and leakage

### 5.1 Canonical key: the first fit's, unchanged

The sha256 of the smaller of the state's serialization and its left-right reflection's (c → 23 − c, bridges
re-canonicalized), with sorted red pegs, black pegs, canonical bridges and side to move (pilot card §4.4). All games
are red-first, so §4.4's colour argument holds as written.

### 5.2 Pools and exact counts

- **Training pool:** one entry per distinct key over the training games' plies 4 to *n*−1, first occurrence in
  (game, ply) order, net of the exclusion sets.
- **Validation pool:** the same over the validation games, keeping only keys that occur **nowhere in the training
  games** (every position, not just selected ones). This dedups mirrored positions across the split.
- **Exact counts: N_TRAIN = 120,000 and N_VAL = 12,000.** Each pool is ordered by
  sha256(`"htb-onpolicy-select/1|<arm>|<key>"`) and the first N are taken. A short pool **REFUSES**, with no fallback.
- **Selection record:** create-only, with each row's game, ply and key, the pool sizes and the exclusion counts. Its
  sha256 binds the labels and the fit.

### 5.3 Exclusion sets (by canonical key; counts recorded)

| Set | Content | Excluded from |
|---|---|---|
| E1 | every position, plies 0 to *n*, of the benchmark's 620 match games and 4 smoke games (#195) | both pools |
| E2 | every position of G3's 220 match games and its smoke pair (#194) | both pools |
| E3 | Diagnostic O's 1,100 positions (#194; inside E2 already, listed for the record) | both pools |
| E4 | the first fit's 264,000 selected rows (240,000 train, 24,000 val) | the validation pool |

E4 keeps the current student's own training positions out of the gate's evaluation set.

## 6. Labels: the first fit's teacher path, with the benchmark's lockstep check

- **Unchanged:** the teacher, its inference path, mirror-averaged targets T(s) and v_T(s) (pilot card §3, §5),
  compact labels as produced (pilot card §16), the shard format and manifests, the non-finite STOP and the refusal
  machinery (pilot card §13.3–13.4).
- **Changed: the lockstep check.** The first label path used the adapter's unfiltered `state_divergences`. Under our
  end rule, twixtbot's early-draw flag fires in long drawn games (G3: plies 73–127; benchmark: 65–163), so that check
  would STOP on a valid game. Labelling uses the benchmark's `bench_divergences` instead: every field compared, the
  adapter's `terminal` line left out, rules-level terminal status compared directly, the flag recorded only. The G3
  draw regression test from the benchmark applies.
- **Legal-set equality** with the teacher at every labelled ply, as before.

## 7. Fit: the first recipe, with the changes listed

| | First H-TB fit (entry 52) | This card |
|---|---|---|
| Architecture | 128 × 6, unchanged inputs | **same** |
| Initialization | `calib020_0001` | **the current student `0e1cb621…`**, digest checked; a second untouched load is the reference arm S |
| Data | red-first C0 corpus positions | **positions from `hard` self-play** (§4–5) |
| Rows | 240,000 / 24,000 | **120,000 / 12,000** (cost; §13) |
| Updates | 2 epochs = 15,000 | **2 epochs = 7,500** (same passes over the data) |
| Selection and epoch namespaces | `htb-select/1`, `htb-fit/1` | `htb-onpolicy-select/1`, `htb-onpolicy-fit/1` |
| Lockstep check in labelling | unfiltered adapter check | the benchmark's check (§6) |
| Device, BatchNorm | MLX CPU, eval mode throughout, running stats bitwise unchanged | **same** |
| Loss | masked CE on both orientations + 1.0 × value MSE + 1e-4 × L2 | **same** |
| Optimizer | Adam lr 1e-4 (defaults read back), clip 1.0, 32 rows × 2 orientations | **same** |
| Start check | KL(T ‖ reference_MA) ≥ 0.01 on 2,048 rows | **same**, the reference now the current student |
| Examples, memory | built per batch; 12 GiB requirement checked at launch | **same** |
| Scoring and gate | F1–F3 vs `calib020_0001` | **I1–I3 vs the current student** (§8) |

## 8. The held-out imitation check: a stop gate

Scored on all 12,000 validation rows with the pilot's mirror-averaged definitions (pilot card §7): KL(T ‖ Q_MA), top-1
agreement, value MSE, for Q = the challenger C and Q = the current student S.

- **I1:** KL_C ≤ **0.90** × KL_S
- **I2:** TOP_C > TOP_S (strict)
- **I3:** MSE_C ≤ MSE_S (no value regression)

**ELIGIBLE** needs I1–I3, every check in §§5–7, and unchanged BatchNorm statistics.
- If every check passes but any of I1–I3 fails, the verdict is **STOP: the challenger missed the imitation gate.**
- A failed check in §§5–7, or changed BatchNorm statistics, is a STOP for that reason, reported as such.

**What is reported either way:**
- all six values: KL_S, KL_C, TOP_S, TOP_C, MSE_S and MSE_C;
- for each metric, C − S, the ratio C / S, and whether it moved toward the teacher, away from it, or not at all;
- which of I1–I3 held.

A missed gate is reported with these movements. It is never summarized as "no improvement" when some metrics moved
toward the teacher.

- **KL_S, TOP_S and MSE_S are the first measurement** of how well the current student agrees with the teacher on
  self-play positions under the research implementation of `hard` (§1). They are reported in their own right.
- **The bars are chosen, not derived:** the 0.90 for I1 and the no-regression bar for I3. A fit that cannot cut its
  KL on these positions by 10% seems unlikely to produce a 0.60 match score.
- The check costs about 15 minutes inside the fit run.

**Descriptive only, never gated:** the same three metrics for C and S on the first fit's 24,000 validation rows, the
labels already in `~/twixt-cache/htb-pilot/labels-val`, verified against their committed manifest. It reports any
forgetting of the old distribution.

## 9. Export and parity, before any game

- Export the challenger once with the **unmodified exporter and the August invocation** used for `78c16a31c3dba37f`.
  It is held outside the product manifest (`~/twixt-cache/htb-onpolicy/`), and `HARD_MODEL_ID` is unchanged.
- **Parity:** the V2-R amendment's surfaces and tolerances, and V3's mirror-averaged parity. Each Python ORT surface
  runs only after the two-conv canary passes (#186). The Node runtime canary (`server/test_ort_conv_canary.js`) passes
  in the run's environment.
- **The graphs must match the current pair's ops and shapes** exactly, so the network cost per evaluation is the same
  by construction.
- Any failure STOPs. The match is not played on an unqualified export.

## 10. The test of improvement: a fresh colour-reversed match against the current `hard`

- **Players:** C's export and `78c16a31c3dba37f`, both through `protocolMove` (200 simulations, T 1.0 through ply 19,
  root halving from ply 20), in one Node process on the product path. This uses the G3 and benchmark driver pattern
  with `process.exitCode` set, never `process.exit()`.
- **Openings:** fresh 4-ply openings, `…/match opening <i>/…`, by the same rule. An opening is skipped if it
  equals, directly or reflected, any of these:
  - a benchmark opening;
  - a **G3 opening** (§4.2: the first four plies of each G3 game);
  - a generation opening or a generation-smoke opening;
  - an earlier match opening.
- **Pairs: 272** (544 games). Pair *p* plays opening *p* twice: game 0 with C as Red, game 1 with C as Black.
- **Seeds:** `sha256("htb-onpolicy/20261114/match/pair <p>/game <g>/ply <k>")`, digits as above. Both players draw from
  the same per-ply seed, as in G3.
- **Pair score:** C's mean over its two games (win 1, draw 0.5, loss 0). Game end by the product's rule only.
- **Pass rule (one-sided Hoeffding, α = 0.05, over pairs):** **PASS** iff mean − √(ln 20 / 544) > 0.5. That is a
  mean above 0.574208: **625/1088 = 0.5744 passes and 624/1088 = 0.5735 does not** (the user verified this boundary).
  Anything else that completes and passes the guards is **NON_PASS**.
- **What a PASS establishes:** the challenger's true mean pair score against the current `hard` is **above 0.50**
  under this protocol, at one-sided α = 0.05.
  - It does **not** establish a score of 0.60 or any other size. 0.60 is the planning target (§11).
  - It holds for **the whole update**: the new data source and the continued training from the current student,
    under this recipe. It does not attribute the gain to either one.
- **Reported, descriptive:** the two-sided 95% Hoeffding interval (half-width 0.0823), results by colour, draws, game
  lengths, and the number of distinct games.
- **Guards (any failure means INVALID, which is not scored):**
  - exactly one row per played ply;
  - every move on both sides records a 200-simulation budget **and** 200 root visits spent;
  - C's total search seconds ≤ 1.10 × the current `hard`'s;
  - both model files' sha256 and the runtime versions checked before load, and again in the record;
  - the run is bound to its launch commit and process group, with the gate restored (benchmark card §7.1).
- **Smoke:** 2 pairs (4 games) in `…/match smoke/…`. They are replay-validated, never read or scored, and used only for
  timing (§12).

## 11. The smallest gain worth pursuing, and the power plan

- **Smallest gain worth pursuing: a true mean pair score of 0.60 for the challenger** (about +70 Elo on the logistic
  scale; a planning conversion, not a claim).
  - **Below that, the test costs more than the experiment.** Detecting 0.55 at 80% power needs about 1,080 pairs
    (about 40 h on the product path), against about 23 h for the generation that produced it.
  - A round of fine-tuning worth less than +70 Elo would not justify an integration cycle (export, parity, G-style
    match) on its own.
- **0.60 is a planning target, not a pass bar.** The pass bar is superiority over 0.50 (§10).
- **Pair count:** the smallest n with about 80% power at 0.60, assuming a pair-score standard deviation of 0.50:
  **272 pairs**.
- **The power figures are model-based.** They come from a normal approximation at an assumed σ, not from a
  distribution-free bound. They are conditional on the match completing and clearing its guards: a VOID or INVALID
  run has no power. They are planning figures:

  | True mean | σ = 0.50 | σ = 0.35 |
  |---|---|---|
  | 0.55 | 21% | 13% |
  | 0.60 | **80%** | 89% |
  | 0.65 | 99% | > 99% |

- **The false-PASS rate at a true 0.5 is at most 5%** for any distribution of pair scores (Hoeffding): the one
  distribution-free guarantee here.
- There is no sequential look, no top-up and no pooling with the smoke.

## 12. Compute guards and ceilings

- **Every stage runs under a wrapper with a fixed ceiling** (TERM 10 s before KILL), a one-line gate-open commit in a
  clean linked worktree, the idle gate, BOINC quit and Unreal Editor closed. The gate is restored on every exit.
- A timeout or failure is **VOID**: no automatic retry, and no partial run is scored or used.

| Stage | Fixed ceiling | Launch rule |
|---|---|---|
| Generation smoke (40 games) | 3,600 s | — |
| Generation | **40 h (144,000 s)**, G_MAX 4,000 | LAUNCH_OK iff Y1–Y5 all hold (§4.4); otherwise REVIEW, with no launch |
| Labelling (132,000 rows) | 6,600 s | 1.5 × the first label run's 32.1 ms/row |
| Fit + gate + descriptive scoring | 11,700 s | 1.5 × (7,500 × 0.708 s + scoring at the first fit's pace) |
| Export + parity | 3,600 s | — |
| Match smoke (4 games) | 1,800 s | — |
| Match | **18 h (64,800 s)** | LAUNCH_OK iff 1.5 × projection ≤ 18 h, projection from the smoke as in the benchmark |

- **Memory:** the fit keeps its 12 GiB launch check. Labelling sets its requirement from the first run's budget,
  scaled to 132,000 rows.
- **Matched compute in the match:** identical graphs (§9), 200 simulations both sides, enforced per move, and the
  1.10 search-seconds guard (§10).

## 13. Total cost (planning estimates from measured paces; not funding)

| Step | Estimate | Pace it rests on |
|---|---|---|
| Generation smoke | ~20 min | 40 games at G3-like lengths |
| Generation | **~23 h** at G3-like lengths (about 3,030 games; the smoke's T = t0 + G_stop × w decides; ceiling 40 h) | 0.536 s/ply on the GPU (entry 54), plus counting and recording |
| Labelling | ~1.2 h | 8,478 s for 264,000 rows (entry 52) |
| Fit, gate and descriptive scoring | ~2.1 h | 0.708 s/update (T-fit); first fit's scoring pace |
| Export and parity | ~0.3 h | V2-R |
| Match smoke | ~5 min | — |
| Match | **~10.2 h** | 1.346 s per searched ply on the product path (benchmark arm P), 50 plies a game |
| **Total** | **~37 h of machine time**, in five separately approved runs | |

- **The main lever is the row count.** Halving it to 60,000 / 6,000 saves about 11.5 h of generation, but leaves the
  fit a quarter of the first fit's updates. I recommend 120,000 / 12,000 as drafted.
- The match size is set by the target and is not a lever.
- Generating on the product path instead would add about 34 h.
- Self-play between equals may draw more often than G3 did. Board-full draws run 572 plies, so the smokes' timing
  checks exist to catch this before the long runs.

## 14. Decision rule (fixed before any work)

| Outcome | Reading | What it permits |
|---|---|---|
| Any stage VOID or INVALID | no result | nothing; no automatic retry |
| Generation smoke **REVIEW** (any of Y1–Y5 fails) | the yield, games or time estimate is low or uncertain | no launch; the plan returns to the user with the numbers |
| Selection refuses a short pool | no result; generation was too short for the counts | a reviewed decision; no silent rescale |
| Gate **STOP: the challenger missed the imitation gate** | the challenger did not clear I1–I3 on the validation positions. Report each metric's movement against the current student. Dose, capacity and labels are not separated | the line closes; no export or match |
| Gate **ELIGIBLE** | the challenger cleared the imitation gate on these positions, and nothing about play | proposing §9 and §10 (approval) |
| Match **PASS** | the updated challenger's true mean pair score against the current `hard` is above 0.50 under this protocol (one-sided, α = 0.05). This covers the whole update (data source and continued training together). It does not establish a gain of 0.60 | proposing an integration decision (export pin, G2/G3-style checks), and nothing more |
| Match **NON_PASS** | superiority not shown. This is not evidence of no gain: the planning power, about 80% at 0.60, is model-based and assumes completion | the current `hard` stays; the line closes; no top-up or re-score |

PASS authorizes no deployment, push or further training. A second on-policy round would need its own card.

## 15. Steps; each needs the user's approval

1. This card and the #53/#87 exceptions.
2. The build: generation driver, the smoke's `yield` analyser (with the §4.4 regressions), selection, the label
   adapter change, fit config, gate scoring, export glue, match driver and analyser. All behind closed gates, with
   model-free tests. Returned once for launch review.
3. Generation: the 40-game smoke and its yield verdict (§4.4), then the run only on LAUNCH_OK. On REVIEW the plan
   returns to the user.
4. Labelling, then the fit with its gate (the fit admits only committed, COMPLETED label evidence).
5. On ELIGIBLE: export and parity.
6. Match: smoke, then the match only on LAUNCH_OK; then the analysis.

Steps 3–4 or 5–6 may be approved together if you prefer.

## 16. Prediction on record

Low-to-moderate confidence, stated before any build:
- **Imitation gate: ELIGIBLE,** with KL_C about 20–40% below KL_S on the validation positions. No
  prediction is offered for KL_S itself.
- **Match: mean pair score about 0.60** (plausible range 0.52–0.70). **PASS is roughly an even chance.**
- **Descriptive retention:** the old-distribution KL changes by less than 10% either way.

## 17. Ledger record (proposal; the long-form entry follows the run)

- **Test:** H-OP, distillation on fresh self-play from the current `hard`: generation (§4), selection (§5), labels
  (§6), fit and gate (§7–8), export and parity (§9), match (§10).
- **Hypothesis:** §1.
- **Why it is plausible:** §1.
  - On-policy labelling targets the student's errors at positions it reaches in play, and the first fit never
    trained on such positions.
  - Arm N (0.045) measured a playing-strength gap against the teacher in raw-policy games. Agreement on `hard`
    self-play positions is an open question; the fit's reference scores measure it first.
  - Search made up little of the gap (P − N +0.070).
  - Imitating this teacher has transferred to play twice (entries 54, 63).
- **Prediction on record:** §16.
- **Decision rule:** §14.
  - Generation smoke Y1–Y5, or REVIEW.
  - Gate I1–I3, or the challenger missed the imitation gate.
  - Match PASS iff mean − √(ln 20 / 544) > 0.5 over 272 pairs: superiority over 0.50 for the whole update; 0.60
    is the planning target.
- **What ran / result / match to prediction / conclusion:** to be written after each judged run.

## 18. What this card does not authorize

Building, generating, loading any model, labelling, training, exporting, playing, opening any gate, deploying or
pushing. Each needs the user's approval (§15). It does not reopen #52, #87 or #53 beyond the requested exceptions, and
it does not use the benchmark's, G3's or Diagnostic O's games for anything but exclusion.
