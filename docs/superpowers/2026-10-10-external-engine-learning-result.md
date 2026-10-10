# External-engine learning — result: PROVISIONAL VIABILITY; strength UNRESOLVED (2026-10-10)

**Card:** `docs/superpowers/2026-10-08-external-engine-learning-viability.md` (design baseline `ff1422fb`).
**Accepted by the reviewer at `92ab3040`.** All three stages ran once each under their own closed gates;
every seed block is closed. Nothing was pushed or promoted.

## Verdict

**Provisional viability** in the card's sense (§stage 3: held-out criteria pass and the screen's interval
does not lie wholly below 0.50). It is **not** promotion, non-inferiority or evidence of stronger play.
**Keep `calib020_0001` as the incumbent.** The stage-2 candidate is retained only as a research artifact.

## What ran

| stage | evidence | result |
|---|---|---|
| 1 — teacher labels | `evidence/external-engine-learning-stage1/` (run `9e3cf468`) | 12 games, 192 rows (128 training / 64 held-out in 4 sealed games); disagreement f = 97/128 |
| 2 — head-only fit | `evidence/external-engine-learning-stage2/` (run `442ab7d8`) | 128 steps; 8 policy-head tensors (519 parameters) changed, the other 84 identical; scalar control beta = 0.5611 fitted on training rows only |
| 3 — held-out + screen | `evidence/external-engine-learning-stage3/` (run `51d58dba`) | SCREEN_COMPLETED, exit 0, 1,983 s of 7,200 s |

Pinned: incumbent `calib020_0001` (SHA-1 `209cf2d4…`); candidate `candidate.safetensors`
(SHA-256 `e73e80b4…`); stage-3 manifest `f26f2f60…`; plan `stage3.json` `0eab28b0…`.

## Held-out imitation — passed, narrowly

Four sealed teacher games, 64 rows, each game weighted equally:

| per-game mean | candidate | scalar control | incumbent |
|---|---|---|---|
| chosen-move cross-entropy | **3.211** | 3.890 | 4.566 |
| teacher top-1 agreement | **25.7%** | — | 20.5% |

The candidate's CE was below both the control's and the incumbent's in every one of the four games, and
its top-1 agreement was never lower than the incumbent's. Beating the control excludes only a global
temperature explanation.

**Limits:** four games from one teacher are not broad generalization. This is closer agreement with the
teacher's move choices, not better play.

## Teacher-rule limitation (accepted for this trial, not removed)

The teacher network (twixtbot) was trained with own-link crossings allowed. Labels were generated on
positions verified legal under our rules, so no unreachable foreign-rule positions were imported. That
does not show the teacher understands our rules or chooses good moves under them. Imitating it is
therefore not evidence of a rules-correct improvement.

## Strength screen — inconclusive

32 of 32 games completed (16 colour-reversed pairs, 400 simulations, no caps, no draws):

* candidate **12 wins / 20 losses**; mean pair score **0.375**;
* unclipped Hoeffding 95% interval **[0.035, 0.715]**; outcome **NOT_VETOED**.

12–20 is an **unfavourable point estimate**. The interval spans 0.50, so it establishes neither inferiority
nor improvement. NOT_VETOED means only that the disaster veto (mean below about 0.1605) did not fire.
Per-colour counts (as red 8/16, as black 4/16) are descriptive only.

## Seeds

All closed by derivation from committed evidence; every started game counted as used.

* stage 1: 18 seeds, all used; exposed and retired (`1d2789ab`);
* stage 2: 1 training seed, used; exposed and retired (`91a9130c`);
* stage 3: 32 game seeds, 32 of 32 started; exposed and retired (`92ab3040`).

The intervals are in the registry (`e4_screen_reference.py`) and the plans; they are not spelled here.

## Open issues — resolve before any future run

1. **Cleanup exit-code ambiguity.** `EL.end_group` (stage 1's process-group cleanup, reused by stages 2 and
   3) lets a `PermissionError` from `os.killpg(pgid, 0)` escape. The parent then writes its cleanup-failed
   record and finalizes nothing, which is correctly fail-closed. But the uncaught exception exits the
   process with status 1, the same code as stage 3's STOP_HELDOUT. Only the cleanup-failed record and the
   absence of `finalized.json` tell them apart. It was observed once, in a stage-1 test under concurrent
   load; it did not occur in any run. Treating `PermissionError` as "group gone" is **not** an acceptable
   fix: cleanup that cannot be verified must stop. The fix needed is a distinct exit status for unverified
   cleanup.
2. **The seed-selection rule is exhausted.** The pinned survey method chooses the next 10,000-aligned base
   above the highest registered seed in the 2026 band. After stage 3 that base is the band's upper end, so any further survey under this method refuses. A new rule (or band) needs its own decision and
   review before any seed accounting.

## Not authorized

No further experiment, rescue dose, beta refit, retraining, checkpoint shopping, enlarged or repeated match,
production expansion, promotion or push.
