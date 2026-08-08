# Competitive Readout — Closeout

**Closed:** 2026-08-08 · **Status:** CLOSED on a NULL. **All frozen outcomes and
consequences are exhausted.** Shipped search, self-play, the checkpoint-tournament
default and `calib020_0001` all stand unchanged.

The line ran every stage its design defined, reached the only benchmark it ever accepted
as decisive, and returned a null. This records what ran, what it establishes, what is
prohibited, and what survives.

## What ran, in order

| stage | outcome |
|---|---|
| **Candidate 1** — all-ply argmax vs shipped tournament readout, 64 games | 54–10, `+293` Elo. **Confirmation only**, no follow-up |
| **Preflight** — frozen rule over Candidate 1's 64 sidecars, read-only | **PASS**, override rate `6.08%`. Reach and scope, **not merit** |
| **Candidate 2 screen** — 64 games | 28–36. Futility **not** triggered; decisive match eligible |
| **Candidate 2 decisive** — 800 games | **408–388, `0.5125`, CI95 `[0.4779, 0.5471]`. Bar NOT met.** |

Seeds consumed: `[202608060, 202608124)`, `[202608124, 202608188)`,
`[202608188, 202608988)`. All three recorded in
`docs/superpowers/2026-08-06-competitive-readout-seed-ledger.md`.

## The decisive result

```
score rate  0.5125    CI95 [0.4779, 0.5471]
LOWER bound 0.4779  →  not above 0.50  →  PROMOTION BAR NOT MET
Elo         +8.7     CI95 [-15.3, +32.8]
```

Colour safety not triggered — red upper bound `0.5874`, black `0.5352`, neither below
`0.50`. No §A integrity abort. Search identity passed (`EXIT=0`) before launch. 796 of
800 decisive, 4 state caps, 13.76 h at `workers=1`.

**This is a NULL, not a harm finding.** The interval spans `−15` to `+33` Elo: the frozen
Hoeffding-LCB override is neither measurably stronger nor measurably weaker than simply
playing the visit leader.

## Forecast calibration, recorded honestly

The 64-game screen lost 28–36 and the authorization set expectations to
"neutral-to-negative." At 800 games the point estimate came back **slightly positive**.

- The **decision-relevant** forecast — that the rule would not clear promotion — was
  **correct**.
- The **point** forecast was **too pessimistic**.

Both belong on the record. The screen's adverse estimate not reproducing is precisely the
irresolution that made the screen futility-only and the decisive match necessary:
**running it was right despite the screen**, and stopping on the screen's point estimate
would have installed a result-dependent gate the design deliberately excluded.

## What this establishes, narrowly

Evidence about a **readout rule** — how a completed search picks its move — at fixed
`calib020_0001`, cold, 400 simulations, in the Python evaluation harness.

It is **not** evidence that the checkpoint is stronger or weaker, and **not** evidence
about the product, which serves a different network entirely
(`MISMATCH`, `docs/superpowers/2026-08-06-model-path-provenance-audit.md`).

Candidate 1's `+293` measured the cost of the shipped tournament readout's deliberate
exploration against a non-sampling opponent. It was a **competitive-configuration
finding**, never a strength claim, and this run does not change that reading.

## Prohibited

Do **not**:

- propose a third readout formula, or any variant of the Hoeffding rule;
- relax the promotion bar, or reinterpret `0.4779` as a near-miss;
- run a larger match to rescue the result;
- mine the 800 captured sidecars for a post-hoc rescue — they are **archival**;
- change shipped search, self-play, the network, the checkpoint-tournament default
  (policy 2), or the product;
- reuse any of the three consumed seed intervals.

The readout was the last untouched axis after tree-local search heuristics closed
(atlas closeout, 2026-08-05). With it null, **search-side work at fixed checkpoint is
exhausted.**

## What survives

**Tooling**, qualified and unspent: `eval_readout.py` (three readout modes and the frozen
rule), `eval_integrity.py` (four fail-closed validators), agent identity decoupled from
checkpoint path, the two-agent match CLI with fail-closed provenance, and the preflight
analyzer. Any successor experiment comparing two agents on one checkpoint can reuse all
of it.

**Process**, now proven under load: authorization-by-countersignature with containment
addressing, a seed ledger with `RESERVED`/`CONSUMED` states, frozen gates written before
telemetry existed, and disposition rules that distinguish integrity aborts from candidate
verdicts. Three runs executed under it without a single post-result rule change.

**Product repairs**, independent of the science: one readout policy across REST and
WebSocket, `deterministicMode` reachable from the transport users actually hit, and a
cache keyed by model and budget with the readout applied after lookup. These stand on
their own as defect fixes and carry **no strength claim** — `MODEL_PATH` provenance is a
`MISMATCH`.

## One process defect worth carrying forward

All three authorizations froze **`--workers 1`**, inherited from the spec rather than
checked against the operator guide, which documents `--workers 4` for exactly this
workload (`docs/post-game-analysis.md:56`, plus the 4,000-game v16 reservoir). The
decisive match therefore took 13.76 h where 4 workers would plausibly have taken 6–10 h.

Every review pass checked that parameters were *pinned*; none checked whether a pinned
value was *right*. **Pinning defends against drift, not against inheriting a bad value.**
Any future authorization that generates games takes its worker count from the operator
guide.

## The pivot

The next project is **not** "train more." It is a small **training-line discovery** that
answers the question left unresolved since June: **is checkpoint strength still improving
beyond the prior training point, or has it plateaued?**

That project needs, before it runs:

- its own baseline;
- an **external strength anchor** — every Elo number in this repository is one sibling
  checkpoint against another, with no fixed reference;
- a deliberately cheap falsification plan, sized so a null costs hours rather than days.

It is a separate scope with its own spec, and nothing in this closeout authorizes it.
