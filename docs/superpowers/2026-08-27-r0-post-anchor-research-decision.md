# R0 — post-anchor research decision gate: **NO_GO**

**Date:** 2026-08-27 · **Status:** DECISION MEMO. Read-only: no implementation, training, model
loading, games, seeds, downloads, or source changes. · Local, unpushed.
**Result: `NO_GO`. No candidate is authorized. `calib020_0001` remains the incumbent and the
research programme stays closed.**

Basis: `main` @ `bce7809`. Authority: `docs/alphazero-value-search-experiment-ledger.md`
(*Status: CLOSED 2026-08-12*), do-not-repeat `#1–#52`, the 2026-08-22 benchmark reset, and the
2026-08-24 C0 card. Evidence: `evidence/2026-08-27-r0-post-anchor-decision/`.

**The `NO_GO` is candidate-driven.** Each candidate fails on its own merits. The benchmark
calculation below prices *future* precision; it does **not** reject anything.

---

## The three candidates

### A. Longer-horizon training — **REJECT: repeats closed mechanisms**

Horizon is a **dose**, not a mechanism, and the ledger closes the families in any dose:

- `#48` — do not "extend the same run to 10 or 20 iterations"
- `#49` — ordinary continuation "in **ANY** dose, warm buffer included — the family is closed"
- `#51` — frozen-parent opposition "in any dose or shape … extend beyond five iterations"

"Not another five-iteration continuation" names a different *number*, not a different mechanism.
There is no cheap falsifier: a horizon effect cannot be falsified short of running the horizon.

### B. Architecture / capacity — **REJECT: no qualified falsifier and no usable corpus**

Not closed by name for the trunk (`#29/#31/#34` concern the v14 *adapter*; `#47` is a *search*
change). It fails on what C0 actually established:

1. **No eligible π corpus.** All **44,900** extant replay games lack the policy target, and
   `GameSaver` cannot write it. π survives on disk in exactly one place — 96 retired atlas legs —
   which cannot serve as a training corpus.
2. **No qualified sensitivity control.** C0's review **struck** width 96 as a positive control:
   fewer parameters guarantee lower nominal capacity, not worse held-out loss.
3. **C0 cannot price this candidate.** Its ≈4.45 GPU-hours/1,000 steps figure covers **only** the
   128×6 final three-seed ladder and excludes the LR grid and checkpoint selection, which C0's own
   review found are themselves training. For a *wider* network the throughput, LR selection,
   checkpoint selection and total step count are all **unmeasured** — calibrating them is what C1
   would have been for. No cost figure for this candidate exists, in either direction.
4. **Unclosed confounds**, and C0 recorded its confidence in the capacity hypothesis as **low**.

> **Withdrawn.** An earlier version of this memo said C0's "cheap falsifier already ran and pointed
> flat", citing `L(96) ≈ L(128)`. **That is false.** C0's scope line reads *"read-only. Nothing was
> trained, loaded, inferred or generated"*, and `L(96) ≈ L(128)` appears there only inside
> conditionals — as the reason 96 cannot serve as a control, not as an observation. No capacity curve
> has ever been measured in this stack. Capacity remains `NO_GO` on 1–4 above, **not** on a
> measurement that does not exist.

### C. Truly independent training data — **REJECT: no qualified source in scope**

Not closed by name. It fails on availability, and the grounds matter:

- **T1j — excluded by construction.** Training on the anchor converts it into a component of the
  thing it measures, irreversibly, with no second uncontaminated anchor to fall back on.
- **twixtbot — excluded on rules provenance, not on `#52`.** `#52` closes *retrying twixtbot as an
  anchor*; it says nothing about using it as training data, and training on twixtbot would **not**
  contaminate T1j. The valid objection is that E0 recorded it as **trained with crossings allowed**,
  while our rules have own-link crossing **off** — its move distribution encodes a different game.
- **Any other source** is outside R0's authorized repository-only scope: it would need a download.
  Nothing in the repository qualifies — the five replay corpora are all self-play and, per C0,
  content-pinned but **provenance-limited**: no game names its generating network.

Stated at its true width: **no qualified uncontaminated data source is available within R0's
authorized scope.** That is a scope-bounded negative, not proof that none exists.

## Decision

**`NO_GO`.** At most one candidate could have been selected; none survives.

| candidate | fails on |
|---|---|
| A · longer-horizon training | repeats closed mechanisms (`#48`, `#49`, `#51`) |
| B · architecture / capacity | no qualified cheap falsifier; no usable π corpus in current assets |
| C · independent data | no qualified uncontaminated source within authorized scope |

**`calib020_0001` remains the incumbent. The research programme stays closed.**

## What a future comparison would cost — precision, not a verdict

Recorded so a future proposal can be priced. A paired comparison plays two checkpoints on identical
opening/colour cells with fresh disjoint seeds; the statistic is the mean per-cell difference,
bounded in [−1, +1]; the method is Hoeffding, as committed in `l0_match_rules`.

**Two different numbers, which an earlier version of this memo conflated:**

| budget | games | pairs | **precision** (CI half-width) | **detectable @ 80% power** |
|---:|---:|---:|---:|---:|
| 6 h | 754 | 377 | 0.1399 | 0.2323 |
| 12 h | 1,509 | 754 | 0.0989 | 0.1643 |
| 24 h | 3,019 | 1,509 | 0.0699 | 0.1161 |
| 48 h | 6,039 | 3,019 | **0.0494** | **0.0821** |
| 96 h | 12,079 | 6,039 | 0.0350 | 0.0580 |

Detection uses the Hoeffding power side — `Δ ≥ t + sqrt(R²·ln(1/β)/(2n))` with `β = 0.20` — and is
**1.66× larger** than the half-width at the same `n`. Per-game cost is **measured**: 28.6 s/game from
the L0 run. One useful incidental: the anchor-referenced benchmark is about **half** the cost of a
native parent match (~60 s/game), because only one side pays the 400-simulation cost.

**What this table does not do.** It does not reject any candidate, and it cannot be compared against
`cont5` (−0.186), `warm5` (−0.068) or `fp6` (−0.034) to argue that future effects are too small to
see. Those are **descriptive** results from different mechanisms, protocols, seeds and evaluation
intervals; the ledger states explicitly that they are **not a paired causal estimate**. No plausible
*positive* effect size has ever been measured in this stack, so none is asserted here.

## What would change this answer

1. **A corpus carrying π**, or a `GameSaver` that writes it — an *engineering* prerequisite, not a
   research result. It is **one** of candidate B's four blockers, not the single one: the missing
   sensitivity control, the invalid cost estimate and the unclosed confounds are independent of it,
   and clearing π alone would not license the experiment.
2. **A named mechanism absent from `#1–#52`** — not a dose, horizon, warmup size, learning rate,
   opponent pool, or intermediate-checkpoint selection of an existing one.
3. **A qualified independent data source** that is neither the anchor nor trained under different
   rules, with a written argument for why using it does not contaminate the external reference.
4. **A preregistered effect size and power target**, priced against the table above — so the
   benchmark is sized before the experiment rather than after seeing its interval.

A proposal that cannot say which of these it satisfies is not a new experiment.

> **A recurring failure mode, named because it accounts for half the corrections below.**
> Five of them are the same error — claiming more than the source supports: a *hypothetical* read as
> a measurement (`L(96) ≈ L(128)`), a ledger entry stretched past its **subject** (`#52` is about the
> anchor pilot, `#49` about continuation from the incumbent), a modal overreach (`cannot` where the
> evidence says *not specified*), a cost **projection** stated as a bound, and a **silently truncated
> scope line** presented as verbatim. Each looked like a citation.
>
> The check that catches all five is one thing: quote the source's scope line **in full**, and confirm
> what the entry is actually *about* — not merely what it is near. The fifth instance is the sharpest
> evidence for it, because it broke that exact rule in the file that records it: **a rule written into
> a document does not enforce itself in the same document.**
>
> A sixth correction is the same family one level down — reporting `wc -c` output as a character
> count, when it measures bytes including the newline. **A tool's number carries the tool's unit, not
> the one you wanted.** Every count in this package is now *computed at generation* with its unit
> named, so it cannot drift from what it describes.

## Corrections carried in this memo

| claim | status |
|---|---|
| "detectable difference" for the CI half-width | **corrected** — precision and 80%-power detection are now separate columns; 0.0494 vs 0.0821 at 48 h |
| "C0's cheap falsifier already ran and pointed flat" | **withdrawn** — C0 trained nothing; `L(96) ≈ L(128)` is hypothetical there |
| candidate C closed by `#52` | **corrected** — `#52` closes twixtbot *as an anchor*; the real objection is rules provenance plus scope |
| "a 64-pair benchmark, the size of the one we just ran" | **corrected** — 64 pairs is **128 games**, twice the L0 run of 64 |
| negative movements used as an effect-size prior | **withdrawn** — descriptive across different mechanisms, explicitly not paired causal |
| capacity needing new self-play "re-enters `#49`" | **withdrawn** — `#49` closes *ordinary continuation from the incumbent*, not from-scratch training of a different architecture |
| a width change "cannot" start from incumbent weights | **corrected** — the supported claim is only that **no qualified widening or initialization method is specified** here |
| π as "the single blocking fact" for candidate B | **corrected** — it is **one of four** independent blockers; clearing it alone would not license the experiment |
| "a wider net from scratch exceeds that, unbounded above" | **withdrawn** — a projection, not a measurement. C0 prices only 128×6; wider throughput, LR selection, checkpoint selection and total steps are unmeasured |
| the ledger's scope line cut at 200, headed "quoted verbatim" | **corrected** — now recorded complete and verified byte-identical; the 700-char truncation notice applies to the numbered entries alone |
| that scope line's length given as "214 chars" | **corrected** — 211 characters, 213 UTF-8 bytes, 214 bytes *with* the LF; `wc -c` measures bytes-with-newline, never characters. All three now computed at generation with units named |
| authorization cites `#1–#51` | the list runs to **`#52`**; twixtbot added one on 2026-08-24, after the 2026-08-12 closure |
