# Combination policy head: fresh-confirmation proposal (paper only)

**Status:** a proposal for the user's decision, 2026-11-02. Nothing here is approved to build or run.

**Recommendation: do not buy another match yet.** Keep the proven combination as the reference. Fund a confirmation
only if a true gain of about 3 percentage points (about 21 Elo) is worth about 30 hours of machine time plus a
segmented-run build. If it is, use design D below; if only 5 points or more would matter, use design C.

## What would be tested

The exact head from the pilot (`head.npz` sha256 `ead156e2…`) against the same frozen proven combination, with the
same search (mirror averaging and root sequential halving from ply 20, 200 simulations) and the same compute guards
(evaluations ≤ 1.05, search seconds ≤ 1.10). It would use colour-reversed pairs on a fresh block. The screen and the
509-pair comparison stay excluded, unpooled and unrescored.

## Costs and power

"PASS at x" is the chance of PASS if the true mean pair score is x.

| Design | Rule | Target | Pairs | Games | Hours | Ceiling (×1.5) | PASS at 0.50 | at 0.52 | at 0.53 | at 0.55 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Identical rerun | Hoeffding | none | 509 | 1,018 | 10.6 | 16 | <0.1% | 0.5% | 3% | 26% |
| A | Hoeffding | 3 points, 50% | 2,050 | 4,100 | 42.7 | 64 | <0.1% | 10% | 50% | >99% |
| B | Hoeffding | 3 points, 80% | 3,036 | 6,072 | 63.2 | 95 | <0.1% | 23% | 80% | >99% |
| B′ | Hoeffding | 3 points, 90% | 3,629 | 7,258 | 75.5 | 113 | <0.1% | 33% | 90% | >99% |
| C | Hoeffding | 5 points, 80% | 1,093 | 2,186 | 22.7 | 34 | <0.1% | 2% | 15% | 80% |
| D | Variance-based | 3 points, 90% | 1,433 | 2,866 | 29.8 | 45 | 2.5% | 58% | 90% | >99% |
| D′ | Variance-based | 3 points, 80% | 1,070 | 2,140 | 22.3 | 33 | 2.5% | 46% | 80% | >99% |
| E | Variance-based | 5 points, 90% | 516 | 1,032 | 10.7 | 16 | 2.5% | 26% | 50% | 90% |

- **Hoeffding rule:** PASS iff the mean pair score − √(ln 40 / (2 × pairs)) > 0.5, plus both guards. This is the
  frozen form of the completed comparison.
- **Variance-based rule:** PASS iff the mean pair score − 1.96 × (sample SD / √pairs) > 0.5, plus both guards.

## Where the numbers come from

- **Pace:** 37.46 s per game, the completed run's whole-run wall time (38,133 s / 1,018 games). The ceiling uses the
  same ×1.5 margin as the completed run. Hours exclude rechecks and closing.
- **Spread:** a pair-score SD of 0.35, from the completed run (sample SD 0.350). The earlier mirror-halving comparison
  gave about 0.31. This is a planning input only; it does not change any completed result, and no planning number
  uses the completed run's mean. The target effects are choices, not estimates from that run.
- **Sizes:** normal approximation. Each row was then checked with 200,000 simulated runs at that row's size, drawing
  from the completed run's pair-score shape tilted to each true mean. If the true SD were the worst case of 0.5, the
  3-point, 80% sizes would grow to 3,517 pairs (Hoeffding, 73 h) and 2,181 pairs (variance-based, 45 h).

## What the choice turns on

1. **The Hoeffding rule is the main cost.** It ignores the actual spread of pair scores, so it needs about 2.5 times
   the games of a variance-based rule for the same power at 3 points. An identical rerun would pass only 3% of the
   time at a true 0.53.
2. **A variance-based rule is a much weaker bar.** Its one-sided false-PASS rate at a true 0.50 is 2.5%, against under
   0.1% for Hoeffding. Adopting it straight after a Hoeffding NON_PASS also looks like moving the goalposts. It would
   need to be frozen before any game, and a PASS under it should not be read as equal to the proven combination's
   Hoeffding PASS. It would also need its own controls, since it assumes the normal approximation holds; the
   simulations above support that only for this shape.
3. **Matches this long are unproven here.** The longest completed mirror-halving or combination match is 10.6 h. A
   kernel panic ended one run (entry 36) and BOINC voided another (entry 39's first strength attempt). Anything over about 12 h therefore needs a new
   segmented design: a frozen schedule, with no scoring until every segment is done and a fixed rule for an interrupted
   segment. Designs at 12 h or less (E, or the identical rerun) could reuse the existing wrapper. Design D would take
   three segments.
4. **Pace is not re-measured.** macOS moved from 27.0.1 to 27.2 after the completed run, and that run slowed from about
   34.5 s per game in its first 500 games to 43.9 s in its last 268; the cause was not measured. A timing check would
   be needed before fixing a ceiling.
5. **A second test should be the last.** Whatever is chosen, I propose no third test of this head, whatever the
   outcome.

Seed room is not a constraint: about 57,000 seeds remain above the last block in the 2027 band, and the rule's gap
floor equals the block size.

## If you approve a design

Each step would need its own approval: the frozen rule and card, the build and its controls, a timing check, the seed
sequence, then the run. This proposal authorizes none of them. No new games, training, deployment or push are
approved.

## Decisions for the user

1. Is a true gain of about 3 points worth buying, or only 5 points or more?
2. If you buy: keep the Hoeffding rule (B or C), or freeze a variance-based rule (D or E)?
