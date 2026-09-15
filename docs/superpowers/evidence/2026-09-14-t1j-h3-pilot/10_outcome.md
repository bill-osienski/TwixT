# H3 PILOT — IT RAN, AND IT COMPLETED

**THE SINGLE AUTHORIZED EXECUTION, 2026-09-15.** `00:47:15Z → 01:14:44Z`,
**27 min 29 s**, wrapper **exit 0**. Every figure below was read from the files
packaged here at the moment of recording.

## 1. THE RESULT

| | |
|---|---|
| games | **40 / 40**, `verdict: OK`, `complete: true`, `timed_out: false` |
| pairs | 20 nominal / 20 scored / **20 distinct** / 0 excluded |
| whole-run limit | 7,200 s CHOSEN; used **1,648.7 s** of it |
| wrapper exit | **0** (`EXIT_COMPLETED`) |

### The stop rules, all frozen before play, all CLEAR

| rule | ceiling | observed |
|---|---|---|
| S1 duplicate pairs | 2 | **0** |
| S2 capped games | 8 | **0** |
| S3 within-pair identical | 2 | **0** |
| S4a total elapsed | 3,600 s | **1,648.7 s** |
| S4b p90 / median | 4.0 | **1.229** |

`interpretation_withheld: false` — the run cleared the 10-pair floor with 20.

### Timing and shape
min **25.9 s**, median **40.5 s**, p90 **49.8 s**, max **70.8 s** per game.
Plies 33 – 88, median 51. **Zero capped games.**

## 2. THE QUESTION THE PILOT EXISTS TO ANSWER

H2 closed because its games did not differ: 14 of its 15 complete cells held
**one distinct game across 46 repetitions**, against a floor of 42.

| | games | DISTINCT transcripts |
|---|---|---|
| **H3 pilot** | 40 | **40** |
| H2 attempt 3 | 692 | 61 |

All 20 pairs contain two different games. **Independently generated openings
produced the variation that repetition did not**, and H2's runaway cap games —
the ones that ate its 28,800 s deadline — are absent entirely.

⚠ This is a statement about DIVERSITY AND RUNTIME, which is what the pilot was
scoped to establish. It is not a statement about strength.

## 3. 🔴 IT IS NOT A STRENGTH VERDICT AND CANNOT BECOME ONE

`is_strength_verdict: false` in the report, by construction. The outcome
distribution over pairs is **reported and decides nothing**:

| ordered outcome (incumbent as red \| as black) | pairs |
|---|---|
| `incumbent_win\|incumbent_win` | 11 |
| `incumbent_win\|incumbent_loss` | 7 |
| `incumbent_loss\|incumbent_loss` | 1 |
| `incumbent_loss\|incumbent_win` | 1 |

No rate, no interval, no comparison is computed from these, and none may be.
The card's permitted outcomes are "authorize design work on a full study" or
"close H3" — and **the pilot cannot authorize a full study automatically.**
That is a separate decision on the evidence, not a consequence of it.

## 4. SEED ACCOUNTING — `[202624000, 202624040)`

```
ACCOUNTED  40      from its registration
EXPOSED    40      every seed carries a COMPLETED game
RETIRED    40      WHOLE BLOCK, one-shot rule
```

40 `task_result`, 40 `opening_bound`, 40 transcripts, and 40 `task_start` each
with a matching `task_done`. **There is no partially-drawn seed to judge** — the
question H2's attempts 2 and 3 each had to answer differently, and the first
time this programme has not had to answer it. Exposure and whole-block
retirement coincide here for the first time, because this is the first one-shot
schedule to COMPLETE.

### 🔴 THE EXPOSURE IS DERIVED, NOT READ
H2's `task_result` carried `seed`. **H3's does not** — `_play_one` returns it and
the run body's projection drops it. The chain is sound and pinned:

* the durable header carries the SEEDED full-field task digest
  `aa527cc9a1a7b1e657911171c63f19fc006909dd64518bd96de3ce4ddfabfba9`, which
  matches `SEEDED_TASK_DIGEST`; and
* `build_tasks` assigns seeds POSITIONALLY, row *i* → `lo + i`.

But it is a **derivation from a pin where H2's was a value in the record**, and
the registry comment says so. A future run should carry the seed in the record.

## 5. THE PROVENANCE REPAIR EARNED ITS KEEP ON ITS FIRST RUN

The durable header carries `selection_mode: "argmax"` and the **whole** incumbent
identity — reference and sha1, eval config, the inert-under-argmax three, the RNG
masks, readout path and agent lifetime — read off the configuration object the
seam handed the builder. Before this session the results file carried two digests
and nothing else.

## 6. 🔴 WHAT WENT WRONG AFTER THE RUN, AND WHAT CAUGHT IT

The retirement made ten tests false, and inverting three of them renamed them.
**The next harness run failed its clean baseline and ABORTED BEFORE THE FIRST
INJECTION**, because five controls still named the old node ids.

That abort is one of the four harness gaps repaired at the start of this session.
Without it, pytest's non-zero exit for a missing node id would have scored those
five controls REJECTED **for free**, and the run would have reported a clean
sweep. **This is the fifth round in which a headline count was not the verdict.**

Two of the five claims were not merely misaimed but **obsolete**:
*"the block is ALSO marked EXPOSED / RETIRED before anything is drawn"* is now
simply true and correct, so the injection would add a duplicate line rather than
a defect. Both were inverted to the live defects — draws left unrecorded, and a
spent one-shot schedule left replayable — exactly as H2's blocks were after
their runs. One re-label collided with H2's existing label and had to be
disambiguated: **one expected reason cannot serve two controls.**

## 7. SCOPE, stated with the result

This establishes, for this design at this size on this configuration: that the
pilot's 40 games complete in ~27 minutes, that every one of them differs, that
every colour-reversed pair differs internally, and that no frozen stop rule
fired. **On this tree, at this commit.**

It establishes nothing about relative strength, nothing about a full study's
size, and nothing that authorizes one. No retry, no replacement block and no
full study is authorized; the push remains held.
