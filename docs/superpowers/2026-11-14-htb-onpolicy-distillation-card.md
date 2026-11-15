# Experiment card H-OP: distillation on fresh self-play from the current `hard`

**Status:** DRAFT for review, 2026-11-14. **Nothing is built, generated, labelled, trained, exported or played under
this card.** It follows the student–teacher benchmark (ledger entry 65, #195), whose arm N result, TRAILS, supports
proposing this experiment. That result authorizes nothing beyond the proposal.

**Approvals this card needs before any work:** the card itself, and the narrow #53 and #87 exceptions in §2. A build
would then return once for launch review. Every runtime step in §15 needs its own approval.

## 1. Question, hypothesis and mechanism

**The specific change.** The first H-TB fit learned from positions in `calib020_0001`'s old self-play corpus. This
card labels positions that the current student reaches **in its own play as `hard`**, using the same teacher, and
continues training the current student on them. Everything else stays as close to the first fit as practical (§7).

**Hypothesis H-OP.** A challenger, warm-started from the current student and fitted to the pinned teacher's
mirror-averaged policy and value at positions from the current `hard`'s own self-play, beats the current `hard` in a
fresh, colour-reversed match under the same 200-simulation protocol, with a true mean pair score of at least **0.60**.

**Why it is plausible:**
- **Distribution shift.** An imitator trained on someone else's positions makes its errors where it actually plays,
  and those errors compound over a game (the standard argument for DAgger-style on-policy labelling). The first fit
  never saw a position from the student's own play, so its play-time positions were never supervised.
- **A large imitation gap remains where it plays.** In the benchmark, the student's raw policy scored **0.045**
  against the raw teacher it imitates (arm N, entry 65). Its top-1 agreement with the teacher was 0.651 on the first
  fit's validation rows (entry 52), and those were not its own positions.
- **Search does not close the gap.** Arm P minus arm N was +0.070 on common openings, so improving the network is the
  lever that remains.
- **Imitating this teacher has transferred to play before.** The first fit's student beat the original proven
  combination by 0.989 (entry 54) and the previous `hard` by 0.932 (G3, entry 63).

**What could make it fail, and what each outcome cannot separate:**
- **Capacity.** The student is 128 channels × 6 blocks, the teacher 48 filters × 20 blocks with 5×5 kernels. The
  remaining gap may be a capacity limit that more data cannot move.
- **The teacher was trained with own-link crossings allowed** (benchmark card §3). Its labels may be mis-specified
  in a way more fitting cannot fix.
- **Diminishing returns.** Closer imitation may no longer buy playing strength at this level.
- **The imitation gate (§8) STOPs:** this cannot separate the dose, capacity and label quality.
- **The gate passes and the match does not:** imitation improved without a gain we can detect. That does not show no
  gain; there is 80% power at 0.60 and much less below it (§11).

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

- **Both sides are the current student under the `hard` protocol**: mirror averaging, 200 simulations, T 1.0 sampling
  through ply 19, root sequential halving from ply 20, the research tie rule.
- **Path (recommended): the research MLX path on the GPU**, in one process, as the H-TB screen and confirmation test
  ran it (`MirrorAveragingEvaluator` + `RootSequentialHalvingMCTS`, student loaded and digest-checked at load and at
  the end, `MLX_ENABLE_TF32=0`).
  - **Why:** measured at 0.536 s per searched ply (entry 54), against 1.346 s on the product's Node path (benchmark
    arm P). That makes about 23 h of generation instead of about 57 h.
  - **Fidelity:** G1 ported this exact protocol to Node, seed for seed (fixtures with ties included), and V2-R/V3
    showed the ONNX pair agrees with this MLX student to about 2.4e-5 with 100% mirror-averaged top-1. Generation needs
    `hard`'s position distribution, not byte equality. The **match** (§10) runs on the product path.
  - The build reuses G1's seed-for-seed fixtures to check the generation driver's move function against
    `protocolMove` with the fake evaluator.
- **Rules and game end:** our rules (red first, no swap, `allow_scl=False`), and the product's end rule only: play
  until a connection, or a draw when no legal moves remain. No ply cap.

### 4.2 Openings, seeds and the split

- **Openings:** each game starts from a supplied 4-ply opening, by the benchmark's rule (benchmark card §6): cells in
  rows and columns 4–19, index `int(sha256("htb-onpolicy/20261114/gen opening <i>/ply <k>")[:12], 16) mod n`.
  - Skipped if it equals, directly or reflected, an earlier generation opening **or any of the benchmark's openings**
    (its 200 match openings and its smoke openings).
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
  (§5.3):
  - in the training games;
  - in the validation games, leaving out any key already seen in a training game.
- **Generation stops** after the first game at which both counts reach **1.15 × N** (138,000 and 13,800), or at
  **G_MAX = 4,000 games**, whichever comes first. The rule reads position counts only, never outcomes.
- Planning figure: about **3,070 games** at 50 searched plies a game (G3 averaged 54.7 plies, benchmark arm P 53.9).
  Reaching G_MAX is not a failure here; a short pool is refused at selection (§5.2).
- **Smoke:** 8 games in a separate namespace (`…/gen smoke/…`), never used for training. Timing only (§12).

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

**ELIGIBLE** needs I1–I3, every check in §§5–7, and unchanged BatchNorm statistics. Otherwise **STOP**. The 0.90 and
the no-regression bar are chosen, not derived. A fit that cannot cut its own-distribution KL by 10% is unlikely to
produce a +0.10 pair score. The check costs about 15 minutes inside the fit run.

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
- **Openings:** fresh 4-ply openings, `…/match opening <i>/…`, by the same rule. Skipped if equal, directly or
  reflected, to any benchmark opening, any generation opening, or an earlier match opening.
- **Pairs: 272** (544 games). Pair *p* plays opening *p* twice: game 0 with C as Red, game 1 with C as Black.
- **Seeds:** `sha256("htb-onpolicy/20261114/match/pair <p>/game <g>/ply <k>")`, digits as above. Both players draw from
  the same per-ply seed, as in G3.
- **Pair score:** C's mean over its two games (win 1, draw 0.5, loss 0). Game end by the product's rule only.
- **Pass rule (one-sided Hoeffding, α = 0.05, over pairs):** **PASS** iff mean − √(ln 20 / 544) > 0.5. That is a
  mean above 0.574208: **625/1088 = 0.5744 passes and 624/1088 = 0.5735 does not.** Anything else that completes and
  passes the guards is **NON_PASS**.
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
- **Pair count:** the smallest n with 80% power at 0.60 under the conservative pair-score standard deviation of
  0.50: **272 pairs**. These are normal-approximation planning figures:

  | True mean | σ = 0.50 | σ = 0.35 |
  |---|---|---|
  | 0.55 | 21% | 13% |
  | 0.60 | **80%** | 89% |
  | 0.65 | 99% | > 99% |

- **The false-PASS rate at a true 0.5 is at most 5%** for any distribution of pair scores (Hoeffding). No sequential
  look, top-up or pooling with the smoke.

## 12. Compute guards and ceilings

- **Every stage runs under a wrapper with a fixed ceiling** (TERM 10 s before KILL), a one-line gate-open commit in a
  clean linked worktree, the idle gate, BOINC quit and Unreal Editor closed. The gate is restored on every exit.
- A timeout or failure is **VOID**: no automatic retry, and no partial run is scored or used.

| Stage | Fixed ceiling | Launch rule |
|---|---|---|
| Generation smoke (8 games) | 1,800 s | — |
| Generation | **40 h (144,000 s)** | LAUNCH_OK iff 1.5 × projection ≤ 40 h, where projection = startup + 3,070 × max(50, smoke mean searched plies) × smoke s/ply |
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
| Generation smoke | ~4 min | — |
| Generation | **~23 h** (G_MAX bound ~30 h) | 0.536 s/ply on the GPU (entry 54), 50 plies a game |
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
| Selection refuses a short pool | no result; generation was too short for the counts | a reviewed decision; no silent rescale |
| Imitation gate **STOP** | this recipe did not improve imitation on `hard`'s own positions; dose, capacity and labels not separated | the line closes; no export or match |
| Gate **ELIGIBLE** | improved imitation on `hard`'s own positions, and nothing about play | proposing §9 and §10 (approval) |
| Match **PASS** | the challenger is stronger than the current `hard` under this protocol (one-sided, α = 0.05) | proposing an integration decision (export pin, G2/G3-style checks), and nothing more |
| Match **NON_PASS** | no gain shown; not evidence of no gain (80% power at 0.60 only) | the current `hard` stays; the line closes; no top-up or re-score |

PASS authorizes no deployment, push or further training. A second on-policy round would need its own card.

## 15. Steps; each needs the user's approval

1. This card and the #53/#87 exceptions.
2. The build: generation driver, selection, label adapter change, fit config, gate scoring, export glue, match driver
   and analyser, all behind closed gates; model-free tests. Returned once for launch review.
3. Generation: smoke, then the run only on LAUNCH_OK.
4. Labelling, then the fit with its gate (the fit admits only committed, COMPLETED label evidence).
5. On ELIGIBLE: export and parity.
6. Match: smoke, then the match only on LAUNCH_OK; then the analysis.

Steps 3–4 or 5–6 may be approved together if you prefer.

## 16. Prediction on record

Low-to-moderate confidence, stated before any build:
- **Imitation gate: ELIGIBLE,** with KL_C about 20–40% below KL_S on `hard`'s own positions.
- **Match: mean pair score about 0.60** (plausible range 0.52–0.70). **PASS is roughly an even chance.**
- **Descriptive retention:** the old-distribution KL changes by less than 10% either way.

## 17. Ledger record (proposal; the long-form entry follows the run)

- **Test:** H-OP, distillation on fresh self-play from the current `hard`: generation (§4), selection (§5), labels
  (§6), fit and gate (§7–8), export and parity (§9), match (§10).
- **Hypothesis:** §1.
- **Why it is plausible:** §1. On-policy labelling targets the student's errors where it actually plays. Arm N
  (0.045) shows a large imitation gap at play-time positions, search closes little of it (P − N +0.070), and
  imitating this teacher has transferred to play twice (entries 54, 63).
- **Prediction on record:** §16.
- **Decision rule:** §14. Gate I1–I3. Match PASS iff mean − √(ln 20 / 544) > 0.5 over 272 pairs.
- **What ran / result / match to prediction / conclusion:** to be written after each judged run.

## 18. What this card does not authorize

Building, generating, loading any model, labelling, training, exporting, playing, opening any gate, deploying or
pushing. Each needs the user's approval (§15). It does not reopen #52, #87 or #53 beyond the requested exceptions, and
it does not use the benchmark's, G3's or Diagnostic O's games for anything but exclusion.
