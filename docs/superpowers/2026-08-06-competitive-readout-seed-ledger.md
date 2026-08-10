# Competitive Readout — Seed Ledger

**Durable record of every seed interval reserved or consumed by this line of work.**

Intervals are **half-open `[start, end)`**: game `g` uses seed `base_seed + g` for
`g in range(games)`, so `end` is the first *unused* seed. `[0, 64)` and `[64, 128)` are
adjacent, not overlapping.

**Every future run in this line must pass all reserved and consumed intervals** via repeated
`--prior-seed-interval START:END`. `eval_integrity.validate_seed_intervals` validates the
whole set pairwise — including priors against each other — and refuses any overlap
**before a single game is played**. Reusing seeds would silently correlate a "fresh" run
with an earlier one, which is invisible in the result and fatal to it.

## Ledger

| # | interval | games | run | date | commit | status |
|---|---|---:|---|---|---|---|
| 1 | `[202608060, 202608124)` | 64 | Candidate 1 — all-ply argmax diagnostic | 2026-08-07 | `d2aaf4f` | **CONSUMED** |
| 2 | `[202608124, 202608188)` | 64 | Candidate 2 — 64-game mechanics screen | 2026-08-07 | `0d9678a` | **CONSUMED** |
| 3 | `[202608188, 202608988)` | 800 | Candidate 2 — 800-game decisive match | 2026-08-08 | `2614795` | **CONSUMED** |

## Flag string for the next run

Copy verbatim. Append one `--prior-seed-interval` per row above.

```
--prior-seed-interval 202608060:202608124
--prior-seed-interval 202608124:202608188
--prior-seed-interval 202608188:202608988
```

The Candidate 2 screen passed only interval 1 as a prior; the decisive match passed
intervals 1 and 2. Any later run in this line must pass all three intervals above.

**The competitive-readout line is CLOSED as of 2026-08-08.** No further interval is
expected to be drawn against it. Any successor project chooses its own range and passes
all three intervals above as priors.

## Successor project — training line

Reserved by the countersigned experiment cards
`docs/superpowers/2026-08-08-training-continuation-experiment-card.md` (row 4) and
`docs/superpowers/2026-08-08-parent-replay-bootstrap-experiment-card.md` (row 5). The
competitive-readout section above remains **CLOSED**; this file is shared only so no
successor picks an overlapping range.

| # | interval | games | run | date | commit | status |
|---|---|---:|---|---|---|---|
| 4 | `[202608988, 202609388)` | 400 | Training continuation — cont5 vs calib020 | 2026-08-08 | `3c70fff` | **CONSUMED** |
| 5 | `[202609388, 202609788)` | 400 | Parent-replay bootstrap — warm5 vs calib020 | 2026-08-09 | `bcf62e2` | **CONSUMED** |
| 6 | `[202609788, 202610188)` | 400 | Frozen-parent opponent — fp5 vs calib020 | 2026-08-10 | `fbe37f3` | **RELEASED UNUSED** |

Row 4 is disjoint from intervals 1–3, which all end at `202608988`; row 5 begins where row 4
ends. **Note:** `eval_checkpoint_match` has no `--prior-seed-interval` and never calls
`validate_seed_intervals`, so disjointness on this path is a manual precondition and this
ledger entry — it is not code-enforced.

Row 4 consumed 2026-08-08 by the countersigned run: 400 games, exit 0, candidate score rate
`0.31375` (CI95 `[0.2687, 0.3588]`), Elo `−136.0`. The promotion bar was **not met**.

Row 5 consumed 2026-08-09 by the countersigned parent-replay bootstrap: 400 games, exit 0,
score rate `0.4325` (CI95 `[0.3843, 0.4807]`), Elo `−47.2`. The promotion bar was **not met**
on the aggregate result. The preregistered absolute per-colour rule mechanically fired for
black (`0.3650`, upper `0.4317`), but the 2026-08-10 ledger erratum makes that result
non-diagnostic of black-specific harm or an independent rejection.

**Ordinary continuation is CLOSED as of 2026-08-09** — self-play continuation from
`calib020_0001` is closed as a family, warm buffer included (do-not-repeat #49). **No further
interval may be drawn for ordinary continuation**, and rows 4 and 5 are its last.

Row 6 is **not** ordinary continuation and does not reopen it: the frozen-parent opponent
changes the data-generating process — the learner plays a fixed checkpoint rather than itself,
and only learner-to-move positions are trained on. It is authorized separately by
`docs/superpowers/2026-08-10-frozen-parent-opponent-experiment-card.md`, which #49 requires to
name a genuinely different mechanism.

**Row 6 is RELEASED UNUSED, by ruling, 2026-08-10.** Its run aborted with exit `134` (a Metal
driver assertion) at the first line of iteration 1, before any checkpoint was produced. The
interval belongs specifically to the **400-game evaluation**, which never started and **drew
zero seeds** — `eval_checkpoint_match` is its only consumer. The authorization is terminated,
so the reservation is released rather than left standing. **A successor may reserve
`[202609788, 202610188)` again, explicitly, in its own countersignature.**

Released **but** spent elsewhere: training seed `20260810` and every `fp5` path are consumed by
the aborted run and must not be reused. Seeds drawn by training come from `--seed`, never from
this ledger.

Any successor passes intervals 1–5 as priors; interval 6 is available.

## Rules

- **RESERVED** means a countersigned authorization owns the interval but execution has
  not begun. Record the reservation before launch so another authorization cannot choose
  an overlapping range.
- **CONSUMED** begins when execution begins. The interval remains spent whether the run
  succeeded, aborted mid-way, or was discarded. A partially consumed interval is still
  unusable: the games that did run correlate any re-use.
- A reservation that was never launched may be released only by an explicit ledger
  entry; silence or an expired plan does not release it.
- Pass every non-released `RESERVED` and `CONSUMED` interval to later runs as a prior.
- Choose the next interval deliberately in its authorization document. Do not pick one at
  run time.
