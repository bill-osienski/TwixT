# H-TB screen plan — the whole student in the proven combination, 32 colour-reversed pairs (2026-11-06)

**Status:** FROZEN at the sha256 pinned in `scripts/GPU/alphazero/htb_screen.py` (`PLAN_DOC_SHA256`). Any edit
means re-pinning under review. The pilot card (`docs/superpowers/2026-11-03-htb-twixtbot-distillation-pilot-card.md`,
sha256 `ccb3bc67…`) is unchanged; this plan fills in its §9.1 screen and §9.3 common rules.

**Authorized by the user's review (2026-11-06):** the minimal build and this paper launch plan only.

**Not authorized:** any real timing, seed procedure (survey, proofs, registration, closing), model loading or game;
opening `HTB_SCREEN_AUTHORIZED`; further training; strength-test work; deployment; pushing. Each step in §9 needs its
own approval.

## 1. What the screen tests, and what it cannot show

- **Question:** does the H-TB student, played whole inside the original proven combination, avoid a gross loss
  against that combination with `calib020_0001`?
- **It is a screen only.** CONTINUE permits proposing the strength test (card §9.2) and nothing else. 32 pairs cannot
  establish stronger play. STOP does not authorize anything.
- **Screen games are never pooled with strength games** (card §9.3).
- **Prediction:** none is made by this plan. Card §12 makes no strength prediction.

## 2. The candidate: the whole student, frozen

- **Identity:** `~/twixt-cache/htb-pilot/student.safetensors`, sha256
  `0e1cb6211aa0b3e2dec6ab6f39bec74ceb8c472c7dcf603e7d331b1d1c1c34e5`, state digest
  `586dca4fcdb87c2611449099b4b7acfda460b13b6717c573a55011b8620a630b`.
- **Bound to the committed fit evidence** (Post-closure entry 52):
  - `evidence/htb-fit/record.json`: `htb-fit-record/1`, COMPLETED, card `ccb3bc67…`, verdict ELIGIBLE with F1, F2 and
    F3 all true; its `student` naming exactly this file, sha256 and digest; its reference digest `ae0e71fc…`; its label
    record hash equal to the committed label record's.
  - `evidence/htb-fit-run/outer.json`: inner and outer exit 0, COMPLETED, gate restored, no survivors, record by script.
- **Loaded whole:** every parameter and every BatchNorm running statistic, through the match's own loader
  (`eval_runner._default_evaluator_factory`), in evaluation mode. No head is installed, and nothing is trained.
- **Frozen:** the file's sha256 is checked before loading. The loaded network's state digest must equal the fit's at
  load **and again at the end**; any difference is a VOID.

## 3. The reference: the original proven combination

- `calib020_0001` (SHA-1 `209cf2d4…`), loaded by the same loader with no head installed. Its state digest must be
  `ae0e71fc…`, the fit's reference digest, at load and at the end.
- The two sides must not share an evaluator or a network, and the candidate's digest must differ from the reference's.

## 4. The match (unchanged from the reviewed screens)

- Both sides: `MirrorAveragingEvaluator` over the loaded evaluator, `RootSequentialHalvingMCTS` at 200 simulations,
  halving from ply 20, on the frozen PROTOCOL (24×24, 280 plies).
- 32 colour-reversed pairs, 64 games, the candidate red in even games; game g plays seed BASE + g, in order, in one
  process, through `mirror_halving_combination.play_and_build`.
- **The loop is the cycle-2 screen's** (`combination_policy_pilot._screen`): `game_start` logged before each game,
  search cost attributed per side and per game, the background-compute check before loading, after every game and
  before the verdict. A test compares the loop's source text with the reviewed one.
- An opening probe at the empty board must find the two sides' priors different.
- Python 3.14.10, MLX 0.30.3, NumPy 2.4.1, checked before any load; `MLX_ENABLE_TF32=0` in the run process only.
- **Device:** the GPU, through that loader, as in every earlier screen. The launch approval must cover GPU games.

## 5. The rule (fixed before any data)

Over the 32 pair scores (a draw or state cap scores 0.5 per game):

- **STOP** if the mean pair score is **below 0.50** (card §9.1). A mean of exactly 0.50 is not below it.
- **STOP** if either measured compute guard fails (card §9.3), each a ratio of the run's totals, candidate over
  reference: **evaluations ratio > 1.05**, or **search-seconds ratio > 1.10**.
- Otherwise **CONTINUE**. The record lists every reason that applied.
- A run that does not finish all 64 games is VOID and receives no score: an interruption, a timeout, a failed check
  or BOINC detected. No automatic retry.

## 6. VOID before any game

The versions; a plan that is not exactly 64 slots from an int base; any seed not ACCOUNTED, available and
schedulable; the checkpoint's SHA-1; the student file's sha256; either digest; shared networks; a module in training
mode; equal opening priors; BOINC or a failed background check.

## 7. Seeds (each step separately approved; none run by this build)

The reviewed two-band method in the 2027 band, by copies of the cycle-2 screen's scripts. The copies differ only in
paths, the plan's kind, module names, titles and verdict lines.

1. **Survey** once (`evidence/htb-screen-seed-survey/01_seed_survey.py survey`): proceed only if CLEAN with every
   control passed.
2. **Pre-registration proof** once: proceed only if it HOLDS.
3. **Registration:** exactly the surveyed block appended last to ACCOUNTED; `evidence/htb-screen-run/plan.json` =
   `build_plan(BASE, survey record sha256)`, kind `htb-screen-plan/1`.
4. **Post-registration proof** once: proceed only if it HOLDS.
5. **After the run, closing:** the seed-usage derivation once (`evidence/htb-screen-seed-closing/01_derive_usage.py`),
   which exposes the started-game prefix and retires the whole block, then the registry edit exactly as derived.

On any failed check: stop and report, without adjusting or repeating the selection.

## 8. The ceiling (PROPOSED for the launch review)

Priced the way the reviewed screens were, from the most recent and largest completed match on this Python
(the cycle-2 strength comparison, Post-closure entry 51: 1,018 games, Python 3.14.10):

- 0.5559 s per candidate search × 63.725 plies per game × 64 games = 2,267.0 s; × 1.5 = 3,400.5 s, rounded up to the
  minute: **3,420 s — TERM at 3,410 s, KILL 10 s later, cleanup observation through 3,430 s.**
- Cross-check: that run's 35.34 s of wall time per game × 64 × 1.5 = 3,392.8 s, inside the same minute.
- Long games: a 280-ply capped game at that run's slowest capped rate (0.6272 s per ply) takes about 175.6 s. With
  the other games at the uncapped estimate, TERM leaves room for 8 capped games of 64, against about 2.8 expected at
  that run's 4.4% cap rate.
- **This is an estimate.** The student plays differently from the reference and from the cycle-2 head; its game
  lengths and search cost are unmeasured. A run the ceiling stops is VOID, with no automatic retry; its progress and
  cause go to the user.

## 9. Launch sequence (paper only; each step needs its own approval)

1. **Seed preparation** (§7, steps 1–4), with routine lifecycle-test updates and focused checks; returned for launch
   review.
2. **Launch review:** the ceiling, the prepared block and the GPU games.
3. **Launch:** pre-launch checks recorded (panic reports, Unreal Editor closed, BOINC quit, versions, checkpoint
   SHA-1, student sha256, fit admission, gates); in a clean linked worktree, a one-line commit opening only
   `HTB_SCREEN_AUTHORIZED`; the wrapper `evidence/htb-screen-run/01_wrapper.sh` once, behind its idle gate and
   background-compute check. Idle waiting is outside the ceiling.
4. **After the run:** the wrapper restores the gate. Evidence is committed only if cleanup is confirmed; an inner
   COMPLETED cannot override an outer TIMEOUT. Then the closing (§7.5), the ledger and routine lifecycle tests.
5. **On any stop, failure, timeout or uncertain cleanup:** the sequence stops and the evidence returns for
   assessment. No automatic retry.
