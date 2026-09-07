# H1 — head-to-head viability screen, PLAN ONLY

**Date:** 2026-08-31 · **Status:** CARD ONLY. **Nothing implemented, nothing executed.**
No model loaded, no JVM started, no game played, **no seed drawn and none registered**,
no gate opened, no training, no push. · Local, unpushed.
The match block is **PAPER-RESERVED here and left UNREGISTERED**, on the §14 convention.

Basis: `main` @ `5e0e99f`, clean, four gates `False`.

---

## 1. The question, and why it comes before D1

**Is `calib020_0001` worth targeted improvement against T1j?**

L0 measured T1j at **0.5938 over 64 games** at `mdPly` 6, Hoeffding 95% **[0.4240, 0.7635]**.
Parity sits inside that interval, so L0 established neither that T1j is stronger nor that it is
not. It also cannot exclude a rout: **0.75 lies inside the interval.**

That matters for sequencing. D1 interrogates positions to find a targetable weakness. If T1j is in
fact winning three games in four, a diagnostic-driven fix is not the right next investment, and D1
would be answering a question about a gap too large for the intervention it feeds. H1 asks the
cheaper prior question and is **decision-relevant either way**: it either clears the path to
targeted work or redirects it.

🔑 **H1 does not replace D1 and does not amend it.** §12/§13/§14 are untouched. If H1 returns
`VIABLE`, D1 remains available on its reserved interval exactly as it stands.

## 2. The design

**8 frozen openings × 2 colour arms × 14 repetitions = 224 games**, all at `mdPly` 6.
112 per colour arm, 28 per opening, **14 per cell across 16 cells**.

**Reciprocal colours, paired openings.** Every opening is played 14 times with T1j as red and 14
with T1j as black. Colour is balanced *within* each opening, not merely in aggregate. This is not
cosmetic: L0 observed `t1j_red` 0.781 against `t1j_black` 0.406, the largest visible effect in the
match. Unbalanced colours within an opening would let that asymmetry leak into the overall rate,
which is the one quantity H1 actually decides on.

**Every shared parameter is READ from L0's frozen plan, never retyped** — openings and their
plies, reference identity, `our_settings` (400 sims), ply cap 280, scoring, per-ply binding, abort
rules, durable format. Retyping eight openings creates a second source that looks identical until
it isn't. A control that inlines them must be rejected.

⚠ **What changes from L0 is the sample size and the seeds, and nothing else.** That is deliberate:
a viability screen that also changed the settings could not be read against L0's measurement.

## 3. Match seeds — a SEPARATE plan, disjoint from §14

**`[202616000, 202616224)`**, half-open, 224 seeds — one per game, seed = `202616000 + task index`
in frozen order. **Reserved on paper only. Not registered, nothing drawn.**

🔴 **The §14 D1 block is in NO registry by design, so a registry-only enumeration cannot see it.**
Reserved-on-paper is still taken. The proof therefore enumerates the registries **plus**
`d1_selection.SEED_INTERVAL` explicitly. Skipping that step would have handed out a block that every
registry check called clean.

- prior seeds enumerated: **3,661** — ACCOUNTED 3,331 · EXPOSED 129 · RETIRED 323 · TEST_ONLY 100 ·
  CONSUMED 1 · **§14 on-paper 221**, deduplicated
- direct overlap: **0** in every category, the §14 block included
- derived-stream collisions: **0**, against each prior seed plus its four masked derivations, both
  colours
- own derivations injective: **True** — **1,120** distinct values from 224 × 5

⚠ **1,120 and the 448 in §7 count different things and neither is a typo.** **1,120** is the
COLLISION-PROOF set: each of 224 seeds plus its four masked derivations, enumerated over both
colours because a task's colour is not fixed at reservation time. **448** is the number of RNG
streams the games ACTUALLY run: two per game — search and readout — for the colour each task is
finally dealt. The proof is deliberately the wider set, so it holds whichever colour a seed draws.
Calling both "streams" without this distinction would recreate the unit ambiguity already corrected
for "45 helper launches" and for §12.5's coincident 1,105.
- exhaustive, not sampled: the widest registered interval is 800 seeds

The proof must be re-run and re-controlled at implementation time against the registries **as they
then stand**, not inherited from this card: a block proved clean today is not proved clean after any
intervening reservation.

## 4. Configurations and timeouts — all frozen here

| | |
|---|---|
| incumbent | `calib020_0001`, 400 simulations, read from L0's sha-pinned plan |
| T1j | `mdPly` 6, pinned `t1j-e1` toolchain, jar and JDK components hash-verified, a root under `/tmp` refused |
| ply cap | 280 (L0's; observed max there was 88, so it is a guard, not a parameter) |
| per-call T1j timeout | **120 s**, §12.10's frozen value |
| whole-run cap | **180 minutes** |
| early stop | **NONE.** `may_stop_early` is `False`, `EARLY_STOP is None` |

**The run cap is set from measurement, not preference.** L0 played 64 games in 30 m 31 s = 28.6 s
per game; 224 games projects to **≈107 minutes**. 180 minutes leaves 69% headroom. Exceeding the
cap is a `VOID`, not a truncated report.

🔴 **No early stop, for the reason L0 recorded.** The canonical screen early-stopped at 7.0/8 =
0.875 and the full 64-game rate came back 0.594. An early stop cannot bias a *band* decision and
does bias a *rate*, and H1's verdict is a rate question. The five screen decision functions must
remain uncallable on an H1 run, with a test asserting each still exists so the prohibition cannot
rot into a reference to nothing.

## 5. The viability threshold — a PREDECLARED DIRECTIONAL RULE on a TWO-SIDED interval

Decided by where **0.75** falls relative to the **two-sided 95% Hoeffding** interval on T1j's
score rate — half-width `sqrt(ln(2/0.05)/(2n))` = `sqrt(ln(40)/448)` = **0.0907** at n=224:

⚠ **The DECISION is directional; the INTERVAL is not.** Only the T1j-dominates direction can return
`NOT VIABLE`, but both bounds come from a two-sided interval at α=0.05, so each bound individually
carries at least 97.5% one-sided coverage. Reading a one-sided rule off a two-sided interval is
CONSERVATIVE, not a coverage claim to bank — and the numbers below are unchanged by saying so.

| verdict | condition | meaning |
|---|---|---|
| **VIABLE** | upper bound **< 0.75** | the gap is not a rout; targeted improvement is worth funding |
| **NOT VIABLE** | lower bound **≥ 0.75** | T1j dominates; a targeted fix is the wrong next investment |
| **INCONCLUSIVE** | the interval straddles 0.75 | 224 games did not resolve it |

**Why 0.75, chosen before any data.** At a 0.75 score rate T1j wins three games in four. Converted
once, for sizing only, that is ≈191 Elo. This project's own interventions have moved strength by
**tens** of Elo, not two hundred: the frozen-parent line closed at −23.5 Elo and the parent-replay
line at −47.2 Elo. A gap of that size is not what "targeted improvement" plausibly closes, so it is
the honest place to put the line.

⚠ **That Elo conversion is design rationale and is a FORBIDDEN CLAIM in the report** (§7). It is
used once, here, before the data exists, to justify where the threshold sits. It may not be used to
describe the result.

**The decisive bands, stated in advance.** Half-width at n=224 is **0.0907**, so:

| observed rate | verdict |
|---|---|
| **< 0.6593** | VIABLE, decisively |
| 0.6593 – 0.8407 (exclusive) | **INCONCLUSIVE** |
| **≥ 0.8407** | NOT VIABLE, decisively |

🔴 **INCONCLUSIVE is a real and reasonably likely outcome, and it is stated now rather than
discovered later.** L0's point estimate of 0.594 sits comfortably in the decisive region and the
design resolves any true rate up to ≈0.659 — but nothing guarantees the rate stays there.

**Why 224 and not fewer or more:**

| n | half-width | decisive VIABLE if observed < | est. runtime |
|---:|---:|---:|---:|
| 64 | 0.1698 | 0.580 | 31 min |
| 128 | 0.1200 | 0.630 | 61 min |
| **224** | **0.0907** | **0.659** | **107 min** |
| 400 | 0.0679 | 0.682 | 191 min |

n=64 is barely decisive against L0's own point estimate and would waste the run. n=400 buys 0.023
of threshold headroom for 84 extra minutes. **224 is the smallest size that stays decisive if the
true rate drifts meaningfully above L0's estimate.**

## 6. VOID and abort rules

🔑 **A COMPLETED GAME IS A RESULT, WHOEVER WINS.** A loss is data. This is the distinction D1
destroyed by conflating `FAIL` with `VOID`, and the reason the low-ply qualification existed.

**`VOID` is instrument failure only:** a T1j timeout, a binder divergence, an illegal move from
either side, a helper transcript that will not parse, a toolchain identity mismatch, a seed outside
the reserved block, or the whole-run cap being exceeded.

- **Any single `VOID` aborts the whole match. There is no partial report.** A truncated match is an
  early stop wearing different clothes, and §4 forbids early stops.
- **The `VOID` must name the position it died on**: task index, opening, colour arm, repetition,
  seed, ply — and carry the helper's own bounded failure transcript. D1's `VOID` named a depth and
  an invocation and nothing else, and the cause was consequently unprovable and remains so.
- **A predeclared, non-analytic `VOID` trace** is written create-only and fsynced per line, on the
  §13 schema-v2 pattern: closed enums, bounded plain ints, exact schema, injected timestamp, no
  free-text and no identity strings. It records how far the run got; it must be structurally
  incapable of carrying a measurement.
- 🔴 **A `VOID` retires `[202616000, 202616224)` WHOLE**, drawn and undrawn alike, on the same rule
  that retired the D1 block. A retry needs a fresh interval, freshly proved, under a fresh
  authorization. **This is expensive on purpose:** a `VOID` at game 220 costs all 224 seeds.

**Cap terminations are RESULTS, not VOIDs** — scored by L0's frozen rule, counted in all 224, and
reported. L0 recorded zero of them with a maximum of 88 plies against a cap of 280.

🔴 **BUT SATURATION IS A THIRD OUTCOME, AND IT IS NOT A VOID EITHER.** If **more than half** of the
games — more than 112 of 224 — terminate at the cap, the match still **plays in full** (caps never
stop it; that would be an early stop), and then reports **`CAP_SATURATED_NO_RATE`: no rate and no
viability verdict.** A rate computed over mostly-unresolved positions measures the cap, not the
players, and a verdict read off such a rate would be a decision about the ply limit wearing H1's
name. This branch is inherited from L0's preregistered rule, applied only *after* all 224 games are
played, and is stated here because an earlier version of this section said cap terminations are
"always scored, counted, and reported" without naming the branch where they stop producing a
verdict.

## 7. Reporting rules

**PRIMARY, and the only quantity carrying a verdict:** T1j's overall score rate over all 224 games,
with a **95% Hoeffding** interval. Closed form, no resampling, no seed to record or lose.

**SECONDARY:** Wilson, reported as **nominal and approximate only**. L0's card recorded why at
length: a previous version called it "provably conservative" on a variance argument, and a variance
inequality is not a coverage statement. Wilson's coverage at n=64 is 94.01% at p=0.5. That correction
is inherited, not re-derived.

**Descriptive only — no interval, no cross-cell comparison, no finding:** per-arm (112 games each),
per-opening (28 each), ply distribution, runtime, cap terminations.

**FORBIDDEN CLAIMS.** L0's list is inherited whole and must live in committed code, not only in this
card, so a reporting script cannot claim what the protocol forbids without editing a module — plus
two additions specific to H1:

- **any pooling of H1 with L0**, or any combined rate across the two matches. They differ in n and
  were preregistered separately; pooling after seeing both is a choice made with the data in hand.
- **any use of the Elo conversion in §5** to describe the result. It sized the threshold; it says
  nothing about what happened.

**The independence caveat is inherited verbatim.** The games are *modelled* as independent. What
verification establishes is that the tasks derive 448 distinct generator streams — two per game,
search and readout — colliding with none used before — that rules out accidental stream **reuse**, not dependence. T1j seeds its own Zobrist
table from an unseeded `Random` per process, which this design neither controls nor observes.

## 8. What this card does NOT do

It implements nothing. **It does NOT register the match block:** `[202616000, 202616224)` is
reserved on paper by §3 and appears in no registry tuple and in no runtime constant. It opens no
gate, loads no model, starts no JVM, plays no game, trains nothing, and pushes nothing.

⚠ **Paper-reserved is not "unreserved", and it is not "registered" either** — it is the §14 state,
and the distinction is load-bearing in both directions. Downward: a block reserved on paper and
never authorized must cost nothing to abandon, which is exactly what made abandoning the D1 block
free. Upward: paper-reserved is still **taken**, so the next reservation must enumerate it
explicitly (§3) even though no registry contains it.

**Separate and unauthorized, each on its own:** implementing H1; **registering** the match block;
opening an H1 execution gate; the run itself; and any training that a `VIABLE` verdict might later
justify.

**If H1 returns `NOT VIABLE`,** the targeted-improvement line stops and the next decision is which
opponent or objective replaces it — not a D1 retry.
**If H1 returns `VIABLE` or `INCONCLUSIVE`,** D1 remains available exactly as §12–§14 leave it.

---

## Attempt 2 — retry preparation (2026-09-07), NOT AUTHORIZED, NOT RUN

Attempt 1 ran once on 2026-09-05 and **VOIDed at game 60 of 224** on the
preference-surface postcondition (attribution unproven); its block
`[202616000, 202616224)` is ACCOUNTED 224 / EXPOSED 60 / RETIRED WHOLE and its
plan `10_h1_plan_v3.json` is **preserved** under its own pins
(`H1_ATTEMPT1_*`). The edited helper was compile-only verified (`89e66ad`) and
runtime-requalified on the eight ply-6 openings (`2f8abd1`, PASS 8/8).

**What changes for attempt 2 — and only this:**

| item | attempt 1 | attempt 2 |
|---|---|---|
| seed block | `[202616000, 202616224)` spent | **`[202617000, 202617224)` PAPER-RESERVED, UNREGISTERED** — collision-proved read-only against every registry, D1's paper block and the spent block: 0 direct, 0 derived-stream collisions over 19,425 prior values, 1,120 own derivations injective; **776-seed gap** from the nearest prior boundary (`retry-prep/03_collision_proof_v3b.py.txt`; the v3 run is kept as the record of a mislabelled control) |
| plan artifact | `10_h1_plan_v3.json` | **`2026-09-07-t1j-h1-retry-prep/06_h1_plan_v4.json`** — built by a recorded script; a test asserts it differs from v3 **only** in `seed_block`, `seed_block_status`, `supersedes`, `shape`, `tasks` (each task: `seed` and derived `rng_streams` only, +1000) and the provenance fields |
| output | `2026-09-05-t1j-h1-match/` | **`2026-09-07-t1j-h1-match-attempt2/01_h1_results.jsonl`, `02_h1_trace.jsonl`**, create-only |
| launcher | ad-hoc script, `sys.exit(0)` unconditional | **`h1_match_command.py`** — reads the runner's gate at both entries; refuses existing output before spawning; runs the match as a worker in its own process group under the shared supervisor (outer cap 180 min + 60 s, SIGINT forwarded, group cleaned after every exit); **restores the gate in the runner source after every exit** and treats a failed restoration as exit 10, superseding all |

**Unchanged, by test:** 224 games = 8 openings × 2 arms × 14 reps at mdPly 6,
ply cap 280, `calib020_0001` at 400 sims, the 0.75 threshold, the decisive
bands, no early stop, the abort / non-abort rules, the estimand, the
independence caveat, the forbidden claims including **no pooling with L0 —
and no pooling with attempt 1's 60 retained games either**, which are a prefix
of a balanced design and not a small version of it.

**Exit codes (one meaning each):** 0 COMPLETED (verdict in the results file) ·
3 VOID · 4 UNEXPECTED · 5 UNAUTHORIZED · 6 TIMEOUT · 7 REFUSED ·
8 CLEANUP_FAILED · 9 INTERRUPTED (drawn seeds stay EXPOSED) ·
10 GATE_NOT_RESTORED.

**Both barriers are UP:** `H1_EXECUTION_AUTHORIZED = False` and the block is in
no registry, so `check_seed_registration` refuses. Execution will need, as
before and in this order: the collision re-proof at that time, registering
ONLY this block, opening ONLY the H1 gate, one run through the wrapper, the
gate restored by the wrapper and verified.
