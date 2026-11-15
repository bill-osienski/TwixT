# H-OP amendment 1 — the generation run's cap (cap only)

- **Date:** 2026-11-15.
- **Amends:** `docs/superpowers/2026-11-14-htb-onpolicy-distillation-card.md`, revision 3 (sha256 `a2a5ba2f7fad3018c16df2abc98c9c5d3ab8e9095415b3c06f41cd35a5aef5f1`). The card's text is unchanged.
- **Status:** the user approved the amendment and its model-free build on 2026-11-15. Launching the generation run needs a separate approval. Every H-OP gate stays closed.
- **Answers:** the generation smoke's yield verdict, **REVIEW, failed Y3 and Y4** (Post-closure entry 66). That verdict and its record stay as written. This amendment does not re-score the smoke, re-derive it under a changed rule, or turn it into LAUNCH_OK.

## 1. The one change

One generation run may play **up to 6,500 games** (G_MAX_RUN) instead of the card's 4,000 (§4.3).

Everything else is unchanged:

| Kept | Value |
|---|---|
| Fixed generation ceiling | 40 h: TERM 143,990 s, KILL 144,000 s |
| Stopping targets | both pools at 1.15 × N: 138,000 training, 13,800 validation positions |
| Row counts | N 120,000 training and 12,000 validation |
| Split | 9:1 by game (game i is validation iff i mod 10 = 9) |
| Smoke rule | Y1–Y5 at G_MAX 4,000; the committed verdict stays REVIEW |
| Exclusions and selection | E1–E4, deduplication and the selection order (card §5) |
| Labels | the pinned teacher, with the benchmark's lockstep check (card §6) |
| Fit recipe | warm start from the current student, the first fit's recipe, 7,500 updates (card §7) |
| Imitation gate | I1 KL_C ≤ 0.90 × KL_S; I2 TOP_C > TOP_S; I3 MSE_C ≤ MSE_S (card §8) |
| Match rule | 272 colour-reversed pairs; PASS iff mean − √(ln 20 / 544) > 0.5 (card §10) |
| Other stage ceilings | label 6,600 s, fit 11,700 s, export 3,600 s, match smoke 1,800 s, match 64,800 s |

The run still stops at the first game after which **both** pools reach their stopping targets. Only the game count at which it otherwise stops moves, from 4,000 to 6,500.

## 2. The extended schedule

- **What is added:** the run needs 6,500 openings, and the frozen list (`docs/superpowers/evidence/htb-onpolicy-openings/openings.json`, sha256 `ca4e3bba…`) holds 4,000. That file is unchanged. The extension `docs/superpowers/evidence/htb-onpolicy-openings/gen-extension.json` (sha256 `578d34f371b238a9919a6eeb9f71c3b30eb286b7b08e95d1722e1fc17f270664`) holds 2,500 fresh openings for run positions 4,000–6,499.
- **The rule** is the card's: the `"gen opening"` sequence under NS `htb-onpolicy/20261114`, continued past the frozen list. Ply k of candidate i takes index `sha256(f"{ns}/gen opening {i}/ply {k}")[:12] mod n` over the sorted legal moves in rows and columns 4–19.
- **Exclusions:** each new opening is distinct, directly or reflected, from:
  - the benchmark's 201 openings;
  - G3's 222 recorded four-ply openings;
  - every frozen list (generation 4,000, generation smoke 40, match 272, match smoke 2);
  - the other new openings.
- **Prefix check:** re-deriving the sequence to 6,500 under the full exclusion set reproduces the frozen 4,000 exactly as its prefix. No scheduled game changes. The new openings are candidate indices 4,000–6,499; none was rejected.
- **Seeds:** the formula is unchanged, `sha256(f"{NS}/gen/game {i}/ply {k}")` with hex digits 0–11 for search and 12–23 for readout. The run's game i uses opening position i and seeds for game i.
- **Uniqueness across all schedules** (generation 6,500 games, generation smoke 40, match 272 pairs × 2, match smoke 2 pairs × 2, every ply 4–575):
  - 4,054,336 seed derivations, all distinct as strings and as (search, readout) pairs;
  - all 8,108,672 48-bit values distinct.
- **Checks:** `htb_onpolicy openings` re-derives both files byte-equal, checks every opening across all schedules and repeats the seed count. The tests do the same. The Node test re-derives the extension with the match analyser's own opening function.

## 3. Admission, bound to the committed smoke

The generation run is admitted only on the smoke of entry 66, exactly as committed. The run's admission checks, in order:

1. **The smoke's wrapper evidence:** a clean, committed completion of the smoke (`stage_admission`, unchanged).
2. **The pinned evidence:** each of these files is committed and equal to its pinned sha256:

   | File | sha256 |
   |---|---|
   | `htb-onpolicy-gen-smoke/record.json` | `f74cc908f8f3b5aa8f4b7a96e377e19eec1a374d7150b25c1d616bf1bd2cb1b0` |
   | `htb-onpolicy-gen-smoke/games.jsonl` | `eb9d2d4e5c60146be7d823de4c379bdec4512f605ba3fbf2de93bd7d0241d286` |
   | `htb-onpolicy-gen-smoke-run/outer.json` | `a99df4307907c479246aae70edba9e60e9cdad29f7a9102c6d869d6f6f5e5847` |
   | `htb-onpolicy-gen-smoke/yield.json` | `b8d8edb98a7b0a5e4b46d2f212fe144b85041d2bec5aa3a11be9d53b17d637b4` |

3. **Re-derivation:** the yield record re-derives byte-equal from the committed smoke under the card's frozen rule.
4. **The REVIEW this amendment answers:** the record is REVIEW, failed exactly Y3 and Y4, at G_MAX 4,000, with the 144,000 s cap and the card's sha.
5. **The amended conditions**, read from that record:
   - Y1, Y2 and Y5 as the record states them;
   - Y3 as G_N ≤ 6,500 and Y4 as G_stop ≤ 6,500;
   - Y5 recomputed as 1.5 × (t0 + G_stop × w) ≤ 144,000 s.

   All five must hold.
6. **The stage's own preconditions:** the amendment file must be the pinned, committed one, and so must the extension.

The run's record carries the amendment's sha256, the yield record's sha256, the cap 6,500, the amended conditions and both time estimates. The label stage also requires the pinned amendment, and validates the run's games against the 6,500-opening schedule.

## 4. Rationale

From the committed yield record (entry 66), 20 games per pool:

| | Training | Validation |
|---|---|---|
| Usable rows per game, u | 26.2 | 26.6 |
| Conservative u_LB | 24.85 | 25.25 |
| Games to the exact counts (G_N) | 5,366 | 4,760 |
| Games to the stopping targets (G_stop) | 5,853 | 5,190 |

- **Why the smoke returned REVIEW:** Y1, Y2 and Y5 held. Only the game cap failed. Exclusions and deduplication removed nothing (f = 1.00), but self-play games are short (mean 30.4 plies), so each game yields about 26 rows.
- **The cap now covers both projections, set by the training pool:** G_N 5,366 is 1,134 games below 6,500, and G_stop 5,853 is 647 below.
  - The thresholds at 6,500: G_N stays within the cap while training u_LB ≥ 20.5, and G_stop while training u ≥ 23.6.
  - At 6,500 games the point estimates give 153,270 training and 17,290 validation rows. The conservative estimates give 145,355 and 16,411. All four exceed N.
- **Time, under the unchanged 40 h ceiling:**
  - At G_stop: T = 80,373 s (22.3 h), and 1.5 × T = 120,559 s.
  - At the cap: 8.374 + 6,500 × 13.730 = 89,256 s (24.8 h), and 1.5 × that = 133,885 s.
  - Both are below 144,000 s, so the ceiling does not bind at the smoke's measured per-game time, even if the run plays to the cap.

## 5. What this does not change or claim

- **Estimates, not guarantees:** these are the smoke's estimates from 20 games per pool, not a guarantee.
- **Stopping at the cap:** a run that reaches 6,500 games without both stopping targets stops at the cap, as the card's run would have stopped at 4,000.
  - Selection then needs N distinct positions in each pool.
  - Short of that, the label stage refuses with no fallback (card §5.2).
- **The ceiling:** a run that the 40 h ceiling stops is VOID, with no automatic retry (card §12).
- **The smoke stays spent:** it is not rerun or extended. Its REVIEW verdict and entry 66 stand as written.
- **One run only:** the amendment covers one generation run. It does not raise the cap again or carry over to any other run.
- **Downstream stages:** label, fit, export and both match stages keep the card's design, thresholds and ceilings. Each still needs its own approval.
