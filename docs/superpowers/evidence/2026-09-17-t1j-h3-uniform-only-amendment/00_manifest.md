# H3 AMENDMENT 3 — UNIFORM-ONLY. Evidence manifest.

**Design only.** Nothing was drawn, registered, generated, or executed. No gate
was opened. The only thing that ran is a collision proof: XOR and set arithmetic
over declared constants, importing the registries and the four masks rather than
retyping them.

| file | what it is |
|---|---|
| `01_collision_proof_v13.py` | the separation proof for the uniform-only generation allocation |
| `02_collision_proof_v13_run.txt` | its run output, verbatim |

## What v13 proves

The study needs one seed per candidate for **296 openings × 400 attempts =
118,400**. The standing uniform range is **59,200** — sized for the 148 openings
of the two-stratum design — and is **short by exactly half**.

Candidate: **`[20261600000, 20261718400)`**, 118,400 wide.

| check | result |
|---|---|
| direct overlap, all ten prior categories | **NONE** |
| derived-stream collisions | **0** of 592,000 derived values |
| injectivity of own derivations | **592,000 = 118,400 × 5** |
| nearest gap to any other reservation | **140,800**, floor **118,400** (its own size) |
| negative controls rejected | **10 / 10** |
| verdict | **CLEAN** |

## 🔴 The term a registry-only enumeration cannot see

Three generation ranges are prior and **no registry holds any of them.** v13 adds
them explicitly — the same correction v10 needed for D1's paper reservation:

* `[20261000000, 20261059200)` — the 148-opening uniform range, **retired unused**
  by this amendment;
* `[20261200000, 20261259200)` — co-produced attempt 1, **SPENT WHOLE** (VOID,
  0 accepted);
* `[20261400000, 20261459200)` — co-produced attempt 2, **SPENT WHOLE** (VOID,
  0 accepted).

Without them the proof would have called an overlapping candidate clean.

## The controls that carry the weight

A check that never rejects proves nothing, so the two boundary cases are the
point:

* a range **starting exactly where attempt 2 ends** — no overlap at all —
  rejects on gap 0 < 118,400;
* a range **one seed inside the floor** rejects on gap 118,399 < 118,400.

The *actual* nearest distance is reported alongside the floor, so a narrow choice
could not hide behind a small threshold.

Control 4 is the naive fix the amendment explicitly declines: extending the
standing range to `[20261000000, 20261118400)` overlaps it by all 59,200 seeds.
It is the same range with a different end.

## Scope

This proves an allocation. It **reserves nothing.** The constant is not in
`h3_study_rules.py` — the code still carries the two-stratum design, and the card
(§9) records that disagreement. Uniform-opening generation, implementation
changes, seed registration, study execution and the push remain unauthorized.
