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

## Flag string for the next run

Copy verbatim. Append one `--prior-seed-interval` per row above.

```
--prior-seed-interval 202608060:202608124
```

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
