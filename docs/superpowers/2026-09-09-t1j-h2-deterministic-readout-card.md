# H2 — Plan-Only Preregistration: is T1j stronger than the **strongest deterministic use** of our current model?

**Status: PLAN ONLY. Nothing here has run.** No code, no runner, no gate, no model load, no JVM, no
game, no seed registration, no training, no confirmation inspection, no push. The implementation and
the execution are **separate authorizations** and neither is requested by this document.

**H2 is not H1 continued.** It asks a different question of a **different incumbent configuration**
with **different seeds**, and §8 forbids pooling or rate-comparison inference with either H1 attempt.

---

## 1. The question, and why it comes before any training

Every measurement so far used the incumbent's **temperature-sampling** readout, in which the played
move is drawn from the visit distribution rather than taken as its maximum. The D1″ reporting
correction established, from the stored histograms, that under that readout a **non-leader move was
played in most positions examined** — some with a visit rank in the hundreds. Whether that costs
playing strength is untested, and it is a cheap thing to change: the same model, the same search, one
different selection rule.

> **PRIMARY QUESTION.** With the incumbent playing **visit-count argmax** after each fixed opening, is
> T1j stronger than our incumbent under otherwise matched conditions?

**Why this is the shortest path.** If T1j is *not* stronger than the deterministic incumbent, the
premise for training against T1j weakens sharply, and the earlier 0.6763 result would be attributable
to the readout rather than to the model. If T1j *is* stronger, we know it beats the strongest
deterministic use of what we already have, and a training question becomes worth designing. Either
answer is worth more than another diagnostic on the same games.

## 2. The design — H1's, with exactly ONE change

| | |
|---|---|
| openings | the same **8**: `o1_center`, `o2_offcenter`, `o3_low`, `o4_high`, `o5_wide_left`, `o6_wide_right`, `o7_diagonal`, `o8_contact` |
| colours | **reciprocal**: every opening played with T1j as red and as black (`t1j_red`, `t1j_black`) |
| T1j | **mdPly 6**, the pinned jar and JDK, unchanged |
| incumbent | `calib020_0001`, **400 simulations**, board 24, batch 14, stall-flush 48 — unchanged |
| ply cap | **280** |
| **THE ONE CHANGE** | incumbent readout `selection_mode`: `opening_temperature` → **`argmax`** |

`eval_readout.MODE_ARGMAX` already exists and is one of the three qualified modes. Under it,
`temp_high`, `temp_low` and `opening_temp_plies` become inert; the card records them as **not
applicable**, never as "unchanged", because a setting that no longer acts is not the same setting.

🔑 **The incumbent identity therefore DIFFERS from L0's and H1's, and must say so.** The recorded
identity must carry `selection_mode: "argmax"`. A run whose identity still reads
`opening_temperature` has not made the change this study is about, and must VOID.

### 2.1 A design check this change makes necessary

T1j is deterministic at fixed depth (E3a: 25 identical queries). With argmax, our readout is
deterministic too. **The remaining source of variation between repetitions of one cell is the search
stream** — the seeded RNG driving prior shuffle and PUCT tie-breaks — which differs per game seed.

If that were not so, all repetitions in a cell would be one game repeated and the effective sample
size would be **16, not 736**. The run must therefore record, and the report must state, the number of
**distinct game transcripts**. **Frozen rule:** if fewer than **90%** of games are distinct, the
primary interval is **not computed** and the outcome is **`INCONCLUSIVE — DEGENERATE DESIGN`**. This
is a validity gate, not a result.

## 3. Sample size, and what it can resolve

| | |
|---|---|
| repetitions per cell | **46** |
| cells | 16 (8 openings × 2 colour arms) |
| **exact task count** | **736 games**, one seed each |
| primary interval | **Hoeffding 95%, two-sided**, distribution-free |
| half-width at n=736 | **±0.0501** |

The score resolves parity only if it falls **below 0.4499** or **above 0.5501** — the interval must
clear 0.50 by its own half-width. That is the honest reach of this design and it is stated **before**
the run: **any score between those two bounds is INCONCLUSIVE**, and no larger n is added afterwards
to rescue it.

**Why 46 and not 14.** H1's 224 games gave ±0.0907, which cannot separate parity from a moderate edge
— exactly the ambiguity that left H1 inconclusive. 736 games buy ±0.05 at a cost of roughly six
hours. Hoeffding is distribution-free, so this reach does not depend on any variance assumption,
including the lower per-game variance argmax is likely to produce.

## 4. The primary rule — parity, frozen

Let **S** = T1j's score (win 1, draw or cap-termination ½, loss 0) divided by 736, and let
**[lo, hi]** be its two-sided Hoeffding 95% interval.

| outcome | condition |
|---|---|
| **T1J_STRONGER** | `lo > 0.50` |
| **NOT_STRONGER** | `hi < 0.50` |
| **INCONCLUSIVE** | the interval contains 0.50 |
| **INCONCLUSIVE — DEGENERATE DESIGN** | distinct games < 90% of 736 (§2.1) |
| **VOID** | any abort, cap breach, identity mismatch, or incomplete schedule |

**One rule, one threshold, decided in advance.** No secondary statistic may override it, and a
narrower interval may not be substituted after seeing the result — **Wilson cannot replace Hoeffding
after the fact**, the specific error H1's closure named.

### 4.1 The 0.75 threshold is REPORTED, and decides nothing here

H1's "worth targeted investment" band at 0.75 may be reported for continuity. **It does not replace
the parity question and cannot change the outcome.** They are different questions: 0.50 asks whether
T1j is stronger at all; 0.75 asked whether it is far enough ahead to justify targeted work.

## 5. Execution bounds

| | |
|---|---|
| whole-run wall clock | **480 minutes**, breached ⇒ VOID |
| per-query timeout | **120 s**, unchanged |
| expected duration | ≈ **5 h 41 m** (H1 attempt 2 measured 27.82 s/game; an ESTIMATE from a different readout, not a promise) |
| cap terminations | scored ½, counted, and reported; a count above 368 (half) is reported as a design warning, not a rule |
| **no early stop** | one preregistered one-shot schedule. **No interim look, no stopping on a lead, no extension.** A partial run is VOID and yields no analysis. |
| retries | **none without a new authorization** |

## 6. Seeds — a fresh interval, PAPER-ONLY

**Proposed:** `[202618000, 202618736)` — 736 seeds, one per game, bound **positionally** (row *i*
carries `202618000 + i`).

**Reserved on paper and REGISTERED NOWHERE by this document.** Registration is a separate, reviewed
edit under its own authorization, so a block that is never authorized costs nothing to abandon.

**The collision-proof procedure, frozen** (to run at registration time, not now):

1. Enumerate every seed in `ACCOUNTED`, `EXPOSED`, `RETIRED`, `TEST_ONLY`, `CONSUMED_SEEDS`, **and
   every paper-only reservation**, each named by its **own constant**, never by an alias.
2. Exclude the candidate from the prior set **by identity** if it is itself already a paper
   reservation — never by subtracting its seeds.
3. Check direct overlap **and** every derived RNG stream (the four frozen masks, both colours).
4. Apply the gap policy: **≥ 224 seeds** from every prior boundary, with the same identity exclusion
   applied to the gap check as to the overlap check.
5. Run **negative controls** that must all reject: each spent block, an interval straddling a boundary
   by one seed, a derived-stream collision with no direct overlap, and the candidate itself without
   the identity exclusion.
6. The enumeration must be **exhaustive**, not sampled, and the widest interval must be small enough
   for that to hold.

## 7. Outputs — locations frozen

Under `docs/superpowers/evidence/<run-date>-t1j-h2-deterministic-readout/`, create-only, with a
`A_MANIFEST.sha256.txt` over every entry and **no footer comment**, so `shasum -c` verifies cleanly:

| file | content |
|---|---|
| `01_h2_plan.json` | the 736 frozen tasks, sha256-pinned |
| `02_prerun_verification.txt` | gates, seed state, identity, output paths absent |
| `03_h2_results.jsonl` | one record per game |
| `04_h2_trace.jsonl` | predeclared event schema, fsynced per line |
| `05_run_command.txt`, `06_stdout.txt`, `07_stderr.txt`, `08_exit.txt` | the invocation and its outcome |
| `09_report.json` | score, interval, outcome, distinct-game count, cap terminations, by-cell table |
| `10_accounting_from_records.txt` | seeds drawn, counted from the records |
| `11_outcome.md` | the verdict and what it does not establish |

## 8. What this may NOT be used for

- **No pooling** with H1 attempt 1 or attempt 2, or with L0. Different readout, different seeds,
  different question.
- **No causal claim from comparing H2's rate with H1's.** The readout differs *and* the seeds differ,
  so any difference is confounded by construction; a comparison may be *displayed* beside its
  confound, never interpreted as the effect of the readout.
- **No training decision.** A `T1J_STRONGER` outcome makes a training question worth *designing*; it
  authorizes nothing.
- **No claim about the readout's effect on our model's strength**, which would need a matched
  argmax-vs-temperature comparison this design does not run.
- **No claim beyond these frozen conditions** — 8 openings, mdPly 6, 400 simulations, ply cap 280.

## 9. Refusals the implementation must carry (for its own authorization)

1. The gate is a reviewed one-line change; the runner reads it and the CLI reads it, and neither
   accepts a flag, an environment variable or a config file.
2. The schedule must be the **exact 736-task frozen plan** by digest; a short schedule must refuse,
   because a budget is a ceiling and bounds nothing below.
3. Every seed **registered** before any is drawn; row *i* carries `202618000 + i`.
4. The recorded incumbent identity must carry `selection_mode: "argmax"`; anything else VOIDs.
5. The gate is restored **immediately and unconditionally** after the process exits, whatever the
   outcome, and process-tree cleanup is verified.
6. Injected-defect controls for each of the above, each proven to reject, over a clean baseline.

## 10. What this document does NOT do

It does not implement, authorize, register, run, or push anything; it does not open confirmation data;
and it does not revisit the closed D1 line, whose verdicts stand.
