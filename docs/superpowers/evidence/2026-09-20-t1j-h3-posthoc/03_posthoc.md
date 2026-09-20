# H3 post-hoc analysis — EXPLORATORY

> 🔴 **Not a verdict, and not preregistered.** Every comparison here was
> chosen after the result was known. Nothing below revises, supports or
> qualifies the study's interval, and no causal claim is made.

The study's answer remains the pinned combined report: **INCUMBENT STRONGER**,
mean 0.7348 over 296 pairs, nominal 95% interval
[0.6559, 0.8137].

**Scope of that claim:** Over legal six-ply TwixT positions drawn UNIFORMLY AT RANDOM and filtered only by the structural admissibility rules of card section 1.3, played in the frozen deterministic argmax configuration. This population says NOTHING about realistic play, about positions either engine would actually reach, or about any engine-produced population. A uniformly random six-ply position is not a position anyone plays.

This script re-derives the mean as 0.734797 from the
sealed records and ABORTS if it disagrees with the pinned report.

## Is the advantage broad or concentrated?

- pairs above parity: **172**, at parity: **97**, below parity: **27** (of 296)
- total excess over parity: **+69.50** pair-points (positive 81.50, negative 12.00)

Trimmed means — dropping the best pairs and re-taking the mean:

| dropped | remaining pairs | mean score |
|---|---|---|
| drop top 5pct (15) | 281 | 0.7206 |
| drop top 10pct (30) | 266 | 0.7049 |
| drop top 25pct (74) | 222 | 0.6464 |

Excess over parity by decile of pair score (decile 1 = strongest):

| decile | pairs | excess |
|---|---|---|
| 1 | 29 | +14.50 |
| 2 | 30 | +15.00 |
| 3 | 29 | +14.50 |
| 4 | 30 | +15.00 |
| 5 | 30 | +15.00 |
| 6 | 29 | +7.50 |
| 7 | 30 | +0.00 |
| 8 | 29 | +0.00 |
| 9 | 30 | +0.00 |
| 10 | 30 | -12.00 |

## Paired outcome categories

| category | pairs |
|---|---|
| won both | 154 |
| split | 97 |
| lost both | 21 |
| won one, drew one | 18 |
| lost one, drew one | 6 |

## By incumbent colour (per game, n=592)

| colour | games | mean points | wins | losses | caps |
|---|---|---|---|---|---|
| red | 296 | 0.7483 | 216 | 69 | 11 |
| black | 296 | 0.7213 | 207 | 76 | 13 |

Difference (red − black): **+0.0270**, Cohen's d +0.063. *Exploratory.*

## By segment (74 pairs each)

| segment | pairs | mean score | sd |
|---|---|---|---|
| 0 | 74 | 0.7432 | 0.3123 |
| 1 | 74 | 0.7027 | 0.2917 |
| 2 | 74 | 0.7736 | 0.3156 |
| 3 | 74 | 0.7196 | 0.3335 |

## Caps

- capped games: **24** of 592; pairs containing a cap: **24**
- mean score, pairs with a cap: **0.6250**; without: **0.7445**
- Cohen's d (capped − uncapped): -0.383. *Exploratory.*

## Game length (plies, per game)

- n 592, mean 63.18, sd 46.87
- min 29.00, q1 45.00, median 52.00, q3 62.00, max 280.00

## Elapsed seconds per game

- n 592, mean 47.93, sd 37.43
- min 17.99, q1 32.42, median 39.69, q3 47.92, max 229.89

## Opening characteristics (geometry only)

Quartile buckets over each feature; mean pair score in each. Geometry is
computed from the six frozen opening stones only — no engine, no rules,
no evaluation. *All exploratory.*

**bbox_area**

| bucket | range | pairs | mean score |
|---|---|---|---|
| 1 | 45.00 – 211.50 | 74 | 0.7331 |
| 2 | 211.50 – 285.50 | 74 | 0.7466 |
| 3 | 285.50 – 340.00 | 82 | 0.7195 |
| 4 | 340.00 – 506.00 | 66 | 0.7424 |

**centrality**

| bucket | range | pairs | mean score |
|---|---|---|---|
| 1 | 4.33 – 7.00 | 76 | 0.7368 |
| 2 | 7.00 – 7.83 | 81 | 0.7407 |
| 3 | 7.83 – 8.50 | 83 | 0.7500 |
| 4 | 8.50 – 9.83 | 56 | 0.7009 |

**elapsed_s**

| bucket | range | pairs | mean score |
|---|---|---|---|
| 1 | 46.46 – 70.52 | 74 | 0.6757 |
| 2 | 70.52 – 80.08 | 74 | 0.7500 |
| 3 | 80.08 – 95.35 | 74 | 0.8041 |
| 4 | 95.35 – 285.82 | 74 | 0.7095 |

**mean_pairwise_distance**

| bucket | range | pairs | mean score |
|---|---|---|---|
| 1 | 6.54 – 10.78 | 74 | 0.7568 |
| 2 | 10.78 – 12.25 | 74 | 0.7061 |
| 3 | 12.25 – 13.61 | 74 | 0.7230 |
| 4 | 13.61 – 16.48 | 74 | 0.7534 |

**min_cross_distance**

| bucket | range | pairs | mean score |
|---|---|---|---|
| 1 | 1.00 – 2.24 | 81 | 0.7469 |
| 2 | 2.24 – 4.06 | 67 | 0.7015 |
| 3 | 4.06 – 6.00 | 76 | 0.7434 |
| 4 | 6.00 – 13.45 | 72 | 0.7431 |

**plies_max**

| bucket | range | pairs | mean score |
|---|---|---|---|
| 1 | 38.00 – 52.00 | 79 | 0.7278 |
| 2 | 52.00 – 60.00 | 70 | 0.7571 |
| 3 | 60.00 – 69.00 | 75 | 0.7467 |
| 4 | 69.00 – 280.00 | 72 | 0.7083 |

## Replay availability

H3's result records carry task_id, pair_id, seed, segment, stratum, opening_digest, transcript_digest, incumbent_colour, winner, terminal_reason, plies and elapsed_s; its transcript records carry only n_plies and opening_bound. NO MOVE LIST WAS PERSISTED, so EXACT H3 VISUAL REPLAY IS UNAVAILABLE and cannot be reconstructed from this evidence. The first SIX plies of every game are recoverable from the frozen population artifact; nothing after ply 6 was recorded. H2 replays exist and may ILLUSTRATE engine behaviour, but they are a different design on a different population and can neither explain nor validate the H3 result.

---

Inputs, by path and sha256, are recorded in `00_inputs.json`; this script
refuses to run if any of them has changed.
