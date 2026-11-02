# Combination policy second cycle: minimal implementation plan and total cost (draft for review, 2026-11-02)

**Status:** a plan only; it authorizes nothing. The science is in
`docs/superpowers/2026-11-02-combination-policy-second-cycle-amendment.md`.

## Total cost

Estimates repeat the first cycle's completed wall times, on the assumption that the teacher's games cost about the same.
Ceilings are indicative, because each is set from new measurements before its approval.

| Stage | Estimate | Indicative ceiling | Basis |
|---|---:|---:|---|
| Generation, 1,000 games | 2.76 h | 6.00 h | 9,929 s completed; the six-hour cap |
| CPU fit, 3,000 updates | 0.52 h | 0.78 h | 1,866 s completed; formula on the first corpus's counts gives 1,865.3 s, ceiling 2,820 s |
| Screen, 64 games | 0.60–0.69 h | 1.03 h | 2,169 s completed; 64 × 64.75 moves × 0.5964 s per search gives 2,471 s, ceiling 3,720 s |
| Strength, 1,018 games | 10.59 h | 14.4–16.4 h | 38,133 s completed; a new screen of 2,169–2,471 s gives ceilings of 51,780–58,980 s |
| **Total** | **about 14.5 h** | **22.2–24.2 h** | |

- **Cumulative stopping points:** about 3.3 h through the fit, and about 3.9–4.0 h through the screen. Strength is the
  bulk of the cost.
- **Seed work:** three blocks of 1,000, 64 and 1,018 seeds, each needing a survey, two proofs, a registration and a
  closing. Each step takes minutes. About 57,000 seeds remain above the last block in the 2027 band, and these need
  about 4,200 including gaps.
- **Disk:** about 314 MB in the row cache and about 40 MB of committed evidence, judged from the first cycle. 210 GB is
  free.
- **Engineering:** about 200 lines of new logic and about 1,000 lines of focused tests, plus about 3,000 lines of
  wrapper, survey and closing copies whose differences are named.
- **The machine:** generation, screen and strength need the GPU and idle gates, with BOINC kept quit by the user.

## What the strength run can resolve

These are model-based estimates, conditional on the run completing and passing the compute guards. They use a normal
approximation at a pair-score SD of 0.35, so they do not establish these rates for every outcome distribution.

| True mean pair score | 0.50 | 0.53 | 0.55 | 0.56 | 0.58 | 0.60 |
|---|---:|---:|---:|---:|---:|---:|
| Chance of PASS | <0.1% | 3% | 26% | 49% | 90% | >99% |

PASS needs an observed mean above 0.5602, about 42 Elo. A second-cycle agent as far ahead as the first head's observed
0.5309 would most likely end NON_PASS and inconclusive again. Under the same assumptions, the screen's STOP fires about
5% of the time at a true 0.50 and about 21% at 0.45.

## Build order

Each row needs its own approval. Build B has to wait for training, because the second head's hash does not exist until
then.

| Step | Work | Runs anything? |
|---|---|---|
| 0 | Review and freeze the amendment; pin its sha256 | No |
| 1 | **Build A:** generation and training kinds, gates, wrappers, one survey and closing copy, tests | CPU tests only |
| 2 | Generation seeds: survey, pre-proof, registration, post-proof | Seed records only |
| 3 | Generation, once; K1 and K2 closing; ledger | GPU, up to 6 h |
| 4 | Fit ceiling from real counts; training, once; ledger | CPU, about 31 min |
| 5 | **Build B:** pin the second head; screen and strength kinds, gates, wrappers, survey and closing copies, tests | CPU tests only |
| 6 | Screen seeds; screen, once; closing; ledger | GPU, about 40 min |
| 7 | Strength seeds; strength, once; closing; ledger | GPU, about 11 h |

## Code changes

All of this goes in `scripts/GPU/alphazero/combination_policy_pilot.py`, in the established pattern of adding kinds.
The four pinned frozen modules, the card and the first pilot's evidence do not change.

**Build A**
- **Kinds:** `cycle2_generate` and `cycle2_train`, each with its own gate, plan, evidence directory and the pinned
  amendment. `run` already refuses a changed amendment, and `conclude` already records its sha256.
- **Paths:** `docs/superpowers/evidence/combination-policy-cycle2-generate` plus `-run`, `-seed-survey` and
  `-seed-closing`; `combination-policy-cycle2-train` plus `-run`; rows under
  `~/twixt-cache/combination-policy-pilot/cycle2_generate/`.
- **Admission:** generation reads the first training run's record and `outer.json` through the existing
  `trained_head`, plus the timing. Training reads the new generation's record and `outer.json`, plus the timing.
- **`_real_gen_seams`:** takes an optional teacher, installs it with `install_trained_head` on the loaded network, and
  checks the digests before wrapping it in `LocalGPUEvaluator`. `_generate` records the binding and runs the probe.
- **`_real_train_seams`:** takes the corpus's kind instead of hard-coding `generate`.
- **`_train`:** takes an optional start. With a teacher, it keeps the parent's value baseline, installs the head,
  requires the teacher's digests in `digests_before` and skips the bit-equal-to-parent check. The first-cycle path
  stays the default.

**Build B**
- **Kinds:** `cycle2_screen` and `cycle2_strength`, again with their own gates and paths.
- **The second head's pin:** `trained_head` takes which pin and records to read, instead of the single `TRAINED_HEAD`.
- **`_screen`:** the literal `kind == "screen"` and `kind == "strength"` branches become fields on the kind, so a kind
  chooses its rule, plan exclusion and record key. The reference stays the parent.

**Copies, with only named differences:** four wrappers, three seed surveys, three closings and four gate modules. The
gate inventory goes from 56 to 60, all closed.

## Focused tests (CPU, fake evaluators, temporary outputs)

- **Teacher binding:** wrong bytes or digests refuse or void generation and training. A tiny real network shows the
  uncompiled evaluator reaches the installed head. Equal probe priors are a VOID.
- **Generation:** the game loop is the same as the first cycle's for the same seed and fake evaluator; slots are
  independent of workers; the rows directory is separate, and an existing one refuses.
- **Fit:** the corpus reader cannot reach the first corpus; the start equals the teacher, with the residual branch
  non-zero; Adam starts empty; the value baseline is the parent's; each STOP rule fires on a constructed case.
- **Preservation:** every first-cycle kind still refuses as one-shot, its evidence bytes are unchanged, and the first
  `head.npz` keeps its sha256.
- **Launch:** fake launches of each new wrapper, with its ceiling re-derived from the committed records; the gate
  inventory; every public entry refuses with its gate closed.
- **Controls:** a few caught mutations on the new branches (the probe, the digest check, the corpus path). Tests hold
  no 2026 or 2027 band seed literal, and pytest keeps the marker exclusion.

## Risks accepted in this plan

- **Inconclusive again:** see the table above; the fixed rule is not reopened here.
- **Self-reinforcement:** a sharper or mistaken teacher could produce sharper or mistaken targets. Entropy, mass on
  first-round eliminations and duplicates are reported, and none of them is a gate.
- **Long games:** the first corpus had 43 state-cap games, taking 664–1,779 s each. The teacher's rate is unknown; the
  six-hour cap bounds it, and a timeout goes to assessment.
- **No comparison between heads:** the design cannot say whether the second cycle beat the first, by choice.
