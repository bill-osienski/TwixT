# H-TB confirmation test plan — the whole student in the proven combination, 110 fresh colour-reversed pairs (2026-11-06)

**Status:** FROZEN at the sha256 pinned in `scripts/GPU/alphazero/htb_strength.py` (`PLAN_DOC_SHA256`). Any edit
means re-pinning under review. The pilot card (`docs/superpowers/2026-11-03-htb-twixtbot-distillation-pilot-card.md`,
sha256 `ccb3bc67…`) is unchanged; this plan fills in its §9.2 strength test and §9.3 common rules.

**Authorized by the user's review (2026-11-06):** the minimal build and this paper launch plan only.

**Not authorized:** any seed procedure (survey, proofs, registration, closing), model loading or game; opening
`HTB_STRENGTH_AUTHORIZED`; further training; deployment; pushing. Each step in §9 needs its own approval.

## 1. What the test asks, and what it cannot show

- **Question:** does the H-TB student, played whole inside the original proven combination, score above one half
  against that combination with `calib020_0001`, at equal measured compute?
- **Standing:** the screen (ledger, Post-closure entry 53: 0.953 over 32 pairs, 61–3) is accepted as strong
  preliminary evidence of a playing advantage against the reference; this is its formal confirmation (card §9.2).
- **Screen games are never pooled with these games** (card §9.3). The screen's block is retired, and the run refuses
  a block that overlaps it.
- **PASS** is a one-sided statement against this reference at these settings: the student in the proven combination
  scores above one half against the original proven combination. It says nothing about other opponents, other
  simulation counts, or which part of the distillation causes the difference (ledger review addendum, 2026-11-06).
- **NON_PASS** is not evidence of no advantage: at a true mean of 0.60 the card's planning estimate gives about 33%
  power.
- **Prediction:** none is made by this plan. Card §12 makes no strength prediction.

## 2. The candidate: the whole student, frozen (unchanged from the screen)

- **Identity:** `~/twixt-cache/htb-pilot/student.safetensors`, sha256
  `0e1cb6211aa0b3e2dec6ab6f39bec74ceb8c472c7dcf603e7d331b1d1c1c34e5`, state digest
  `586dca4fcdb87c2611449099b4b7acfda460b13b6717c573a55011b8620a630b` — the screen's own constants
  (`htb_screen.STUDENT`), imported, not retyped.
- **Loaded whole** through the match's loader (`eval_runner._default_evaluator_factory`), in evaluation mode. No head
  is installed, and nothing is trained.
- **Frozen:** the file's sha256 is checked before loading; the loaded network's state digest must equal the fit's at
  load **and again at the end**; any difference is a VOID.

## 3. The reference: the original proven combination (unchanged from the screen)

- `calib020_0001` (SHA-1 `209cf2d4…`), loaded by the same loader with no head installed; state digest `ae0e71fc…` at
  load and at the end. The two sides must not share an evaluator or a network.

## 4. Admission (refused before anything is loaded)

- **The fit:** the screen's admission, unchanged (`htb_screen.fit_admission`): the committed fit record ELIGIBLE with
  F1–F3 true, naming exactly the pinned student and reference digest, and its outer.json COMPLETED with cleanup
  confirmed.
- **The screen:** the committed `evidence/htb-screen/record.json` must be `htb-screen-record/1`, COMPLETED, verdict
  CONTINUE (and its `screen.verdict` CONTINUE), under the pinned card and the screen plan's pin, naming exactly the
  pinned student, with both digests the pinned ones at load and at the end; and `evidence/htb-screen-run/outer.json`
  inner and outer exit 0, COMPLETED, gate restored, no survivors, record by script.

## 5. The match (unchanged from the screen)

- Both sides: `MirrorAveragingEvaluator` over the loaded evaluator, `RootSequentialHalvingMCTS` at 200 simulations,
  halving from ply 20, on the frozen PROTOCOL (24×24, 280 plies).
- **110 colour-reversed pairs, 220 games**, the candidate red in even games; game g plays seed BASE + g, in order, in
  one process, through `mirror_halving_combination.play_and_build`.
- **The loop is the screen's, which is the cycle-2 screen's** (`combination_policy_pilot._screen`): `game_start`
  logged before each game, search cost attributed per side and per game, the background-compute check before
  loading, after every game and before the verdict. A test compares the loop's source text with both.
- An opening probe at the empty board must find the two sides' priors different.
- Python 3.14.10, MLX 0.30.3, NumPy 2.4.1, checked before any load; `MLX_ENABLE_TF32=0` in the run process only.
- **Device:** the GPU, through that loader. The launch approval must cover GPU games.

## 6. The rule (card §9.2, fixed before any data)

Over the 110 pair scores (a draw or state cap scores 0.5 per game), in float64:

- **lower = mean − √(ln 20 / 220)**, the one-sided Hoeffding bound at α = 0.05 for 110 scores in [0, 1]. The
  half-width is 0.116692; lower > 0.5 means a mean above **0.616692**. Reachable means are multiples of 1/220:
  136/220 = 0.6182 passes and 135/220 = 0.6136 does not.
- **PASS** only if lower > 0.5 **and** the evaluations ratio ≤ **1.05** **and** the search-seconds ratio ≤ **1.10**
  (ratios of the run's totals, candidate over reference). The evaluations guard is unchanged despite the screen's
  narrow margin (1.0460).
- Otherwise **NON_PASS**. The record lists every reason that applied.
- This is **not** `mirror_halving_combination.hoeffding_lower`, which uses ln(2/α) (ln 40 here, half-width 0.129490).
  A test pins the difference.
- A run that does not finish all 220 games is VOID and receives no score: an interruption, a timeout, a failed check
  or BOINC detected. No partial scoring and no automatic retry.

## 7. VOID before any game

The versions; a plan that is not exactly 220 slots of `htb-strength-plan/1` from an int base; any seed not
ACCOUNTED, available and schedulable; a block that overlaps the screen's (read from the screen's committed plan); the
checkpoint's SHA-1; the student file's sha256; either digest; shared networks; a module in training mode; equal
opening priors; BOINC or a failed background check.

## 8. Seeds (each step separately approved; none run by this build)

The reviewed two-band method in the 2027 band (gap floor = the block's own size, 220), by copies of the H-TB screen's
scripts. The copies differ only in the block size (220), the paths, the plan's kind, module names, titles and verdict
lines.

1. **Survey** once (`evidence/htb-strength-seed-survey/01_seed_survey.py survey`): proceed only if CLEAN with every
   control passed.
2. **Pre-registration proof** once: proceed only if it HOLDS.
3. **Registration:** exactly the surveyed block appended last to ACCOUNTED; `evidence/htb-strength-run/plan.json` =
   `build_plan(BASE, survey record sha256)`, kind `htb-strength-plan/1`.
4. **Post-registration proof** once: proceed only if it HOLDS.
5. **After the run, closing:** the seed-usage derivation once (`evidence/htb-strength-seed-closing/01_derive_usage.py`),
   which exposes the started-game prefix and retires the whole block, then the registry edit exactly as derived.

On any failed check: stop and report, without adjusting or repeating the selection.

## 9. The ceiling (PROPOSED for the launch review)

From the screen's measured pace (the same loader, loop, sides, settings and machine; ledger entry 53), as the
cycle-2 strength ceiling was priced from its screen:

- The screen's wall time 1,279 s × 220 / 64 × 1.5 = 6,594.8 s, rounded up to the minute: **6,600 s — TERM at
  6,590 s, KILL 10 s later, cleanup observation through 6,610 s.**
- Cross-checks: the screen's summed game time (1,273.9 s) gives 6,568.3 s, and its record's total (1,278.6 s) gives
  6,593.0 s, both inside the same minute.
- Long games: the screen had no capped game. A 280-ply game at the screen's slowest per-ply rate (0.5492 s) takes
  about 153.8 s, or 175.6 s at the cycle-2 strength comparison's slowest capped rate (0.6272 s). With the other games
  at the screen's pace (19.98 s), TERM leaves room for 16 or 14 such games of the 220.
- **This is an estimate, and its main risk is game length.** The screen's games were short (37.1 plies on average,
  27–66) because the student won 61 of 64. TERM covers an average of 29.95 s per game, 1.5 times the screen's pace,
  or about 56 plies per game at the screen's 0.536 s per ply. If these games run as long as the cycle-2 strength
  comparison's (63.7 plies, 35.3 s per game), the run reaches TERM and is VOID. A run the ceiling stops is VOID, with
  no automatic retry; its progress and cause go to the user.
- Idle waiting (the idle gate) is outside the ceiling.

## 10. Launch sequence (paper only; each step needs its own approval)

1. **Seed preparation** (§8, steps 1–4), with routine lifecycle-test updates and focused checks; returned for launch
   review.
2. **Launch review:** the ceiling, the prepared block and the GPU games.
3. **Launch:** pre-launch checks recorded (panic reports, Unreal Editor closed, BOINC quit, versions, checkpoint
   SHA-1, student sha256, fit and screen admission, registry, gates); in a clean linked worktree, a one-line commit
   opening only `HTB_STRENGTH_AUTHORIZED`; the wrapper `evidence/htb-strength-run/01_wrapper.sh` once, behind its
   idle gate and background-compute check.
4. **After the run:** the wrapper restores the gate. Evidence is committed only if cleanup is confirmed; an inner
   COMPLETED cannot override an outer TIMEOUT. Then the closing (§8.5), the ledger and routine lifecycle tests.
5. **On any stop, failure, timeout or uncertain cleanup:** the sequence stops and the evidence returns for
   assessment. No automatic retry.
