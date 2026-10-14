# Search distillation — Stage B closeout (2026-10-14)

**Verdict: NON-PASS.** Q held, L1 held, L2 and L3 failed. The pilot is closed. `calib020_0001` remains
the incumbent. There is no strength evidence in either direction: no strength game was played.

Card: `2026-10-12-search-distillation-stage-b-card.md` (rev 2). Evidence:
`evidence/search-distillation-stage-b/{G,P,L1,Q,L2,T,K}/`, all committed on
`codex/search-distillation-stage-b`.

## 1. The question (unchanged from the card)

On held-out games, does training the incumbent's eight policy-head tensors on 1,600-simulation visit
distributions predict held-out 1,600 distributions better than (a) the same procedure on 400-simulation
targets, (b) the unchanged incumbent, and (c) the incumbent with one fitted temperature?

## 2. Fixed design (as run, nothing tuned)

* **Corpus:** 200 fresh games, incumbent at 400 vs itself at 400, 24×24, 280-ply cap.
* **Split:** 40 held-out games (g % 5 == 4) and 160 training games.
* **Positions:** up to 4 per game, selected before any label.
* **Targets:** root visits from `search_with_root(state, add_noise=False)` over the full legal list, at 1,600
  (student) and 400 (control), sharing a label seed.
* **Training:** exactly the eight policy-head tensors (519 parameters), BN running statistics frozen, eval
  mode. 160 steps × batch 32, Adam lr 1e-3, clip 1.0, one shared index schedule for both arms, final step
  only.
* **Temperature arm:** one β fitted on the training positions' T1600 targets.
* **Criteria:** L1 and L2 each need ≥ 26 of 40 held-out game wins (ties and empty games lose); L3 needs
  student top-1 agreement strictly above both control and incumbent.

## 3. Results

**Phases.** Each ran once, with outer exit 0, no survivors and `COMPLETED`.

| Phase | Commit | Wall time | Output |
|---|---|---|---|
| G | 414afe5e | 40,997 s (cap 43,190) | 200 games; result-set check passed |
| P | ec639584 | 1 s | 800 selected; held out: 152 kept, 8 dropped by leakage, 0 empty games |
| L1 | a02f8a9a | 2,160 s | 160 held-out labels; check D passed in all 40 games |
| Q | 1358bb0e | 1 s | QUALIFIED |
| L2 | 2ccc7fca | 5,208 s | 640 training labels; check D passed in all 160 games |
| T | 30da129a | 27 s | both arms trained; verdict NON-PASS |
| K | 34b6cd2f | 0 s | whole block exposed and retired |

**Gate Q** (152 retained positions):
* Q1: 40/40.
* Q2: sum R 0.0009375 / sum D 15.263 = 6.1e-05 (limit 0.5).
* Median D 0.389; median R 0.0; repeat top-1 agreement 1.0.

**Learning criteria** (40 held-out games, no empty games):

| Criterion | Student wins | Required | Result |
|---|---|---|---|
| L1: KL_student < KL_control | 27 / 40 | ≥ 26 | holds |
| L2: KL_student < min(KL_incumbent, KL_temperature) | **18 / 40** | ≥ 26 | **fails** |
| L3: top-1 strictly above control and incumbent | 0.140 vs 0.158 / 0.181 | — | **fails** |

**Corrected interpretation of L2.** 18/40 is the **joint** comparison: a game counts only if the student
beats both the incumbent and the temperature arm. Taken individually, from the committed per-game results:
* student vs temperature alone: **25/40**;
* student vs incumbent alone: **25/40**.

Neither individual count reaches 26. The frozen L2 criterion is the joint one, and it fails.

**Descriptive** (not criteria):

| Arm | Mean KL | Top-1 agreement |
|---|---|---|
| student (T1600) | 1.5096 | 0.1396 |
| control (T400) | 1.5651 | 0.1583 |
| temperature (β = 0.7165) | 1.5561 | 0.1813 |
| incumbent | 1.6708 | 0.1813 |

* Relative KL gain of the student over the incumbent: 9.6%.
* Surface check: in both candidates exactly the eight head tensors changed and the other 84 of 92 saved
  tensors are byte-identical, including the BN running statistics.

**Seeds.** K derived the whole Stage B block as used. Registry closing edit 6299beaf appended it last to
EXPOSED and RETIRED. ACCOUNTED is unchanged. `tests/test_seed_selection.py`: the original run gave
17 passed with exit status unknown (not captured); one additional authorized run gave 17 passed, exit 0.

## 4. What the result says

* The recipe improved distribution fit. The student has the lowest mean KL of the four arms, and it beats
  the 400-target control in 27 of 40 games.
* It did not meet the viability requirement. It does not beat the better of incumbent and temperature in
  enough games (18/40 joint; 25/40 against each alone).
* It lost top-1 agreement: 0.140 against the incumbent's 0.181. Both trained arms lost top-1 relative to
  the incumbent.
* A single fitted temperature recovers much of the KL gain (1.556 against the student's 1.510) without
  changing top-1.

## 5. Limitations

* **No strength evidence.** No game was played with either candidate. Gameplay remains unmeasured, and
  this closeout makes no strength claim in either direction.
* **One recipe.** lr, step count, positions per game, corpus size and the temperature arm were all fixed
  design choices. A non-pass says nothing about other recipes, and the card allows no automatic expansion.
* **Q's margin is uninformative about target quality.** R ≈ 0 means the 1,600 search barely depends on its
  seed. On the tested positions and under this protocol, Q shows the targets are reproducible and differ
  from the 400 targets. It does not show that T1600 is a better target, nor that the same holds elsewhere.
* **Independence is assumed.** The sign tests treat the 40 held-out games as independent.
* **Corpus is incumbent 400-v-400 play.** A student would meet other positions in its own play.
* **The held-out positions are spent.** They must not be used to retune this recipe.

## 6. Decision

* **Candidates preserved.** Both are kept outside git at their recorded paths and SHA-256s:
  * `runs/search-distillation-stage-b/stage-b-student.safetensors` (`701637796571679b…`);
  * `runs/search-distillation-stage-b/stage-b-control.safetensors` (`db882c22c292f7dc…`).
* **Incumbent unchanged.** `calib020_0001` remains the incumbent; neither candidate replaces it.
* **Held-out material spent.** The Stage B held-out games and positions must not be reused to tune or
  re-evaluate this or a related recipe.
* **No further experiment authorized.** Stage A demonstrated stronger search. This fixed training recipe
  improved average target fit but failed Stage B's viability criteria. No model-strength improvement was
  established. Any further work needs its own card and approval.
