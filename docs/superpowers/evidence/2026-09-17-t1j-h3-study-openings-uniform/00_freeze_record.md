# H3 POPULATION FREEZE — RAN ONCE, 2026-09-17. COMPLETED.

Authorized as a single attempt. Run once, not retried.

    .venv/bin/python -m scripts.GPU.alphazero.h3_freeze_command --run

| | |
|---|---|
| outcome | **COMPLETED** |
| exit | **0** |
| barrier readback (from the FILE) | **`False`** |
| terminal verdict | **OK** |
| accepted / expected | **296 / 296** |
| elapsed | **0.667 s** |
| attempts total | **296** (one per opening; rejection rate 0.0000) |
| `opening_set_digest` | `35932b3fabd9c6463d615b0b3af380134dadd700e2ca1e882a0c863faf772e46` |
| range retired | **none** — engine-free generation is deterministic and repeatable |
| surviving processes | **0** (no model, no JVM, nothing to contain) |

## Preflight, immediately before

`h3_study_prerun_verification` on a clean tree at `725a285`: **exit 0, 0 FAILED,
85 PASS, 2 PENDING BY DESIGN** — the population, and the seed block.

⚠ The preflight had been **crashing mid-run since the uniform rewrite** on two
dead references (`stub`, `match_cfg`), so everything after the real-builder
section — the seed block, the output paths, the wrapper, the verdict — never ran,
and the exit code was 1. Repaired at `90bd18b`/`725a285` before this attempt. **A
checker that dies mid-way looks exactly like one that finished**, which is why
reading its tail was not enough.

## The population

* **296 openings, all distinct** up to the board's symmetry group; indices
  0..295 contiguous; segments 74/74/74/74.
* every opening **replayed through the real engine** and its canonical digest
  **recomputed**, not read; every `seed == attempt_seed(base, index, attempts−1)`;
  every candidate **re-derived from the PRNG** and matching.
* seeds `20261600000 .. 20261718000`, all inside the proved range
  `[20261600000, 20261718400)`, all distinct.
* **zero overlap** with the 28 excluded positions (20 pilot + 8 H1/H2).
* no `order` field anywhere; one stratum, `uniform`.

## Generator identity

Engine-free. **Enforced**: `kind`, `engine_free`, `gen_seed_base`, `seed_range`,
`max_attempts`, `opening_plies`, `board_size`, `n_pairs`, `filters`,
`bit_generator` (PCG64), `source_pins`, `excluded_digest_set`, `numpy` (2.4.1).
**Recorded only**: `python` (3.14.7), `commit` (`725a2857…`).

Pinned sources — and *only* these, which is the repair that makes the two-step
sequence possible:

    h3_generation_protocol.py   f371cf1817c7272cc7d7443952a8ad0a…
    game/twixt_state.py         291275223b71b522a19181ae33a34518…
    d1_selection.py             ebea796093bdd7cbecb17ce719df5040…

`h3_study_rules.py` is **deliberately not pinned**: the next step edits it, and
pinning it made the population invalidate itself on its own required next step.
Verified here — that file's hash moves when the digest is recorded, and the
artifact still validates.

## What this is NOT

**`OPENING_SET_DIGEST` is UNSET and the study cannot run.** The runner refuses an
unpinned population, which it demonstrably does. Freezing produced the
population; **recording its digest is a separate reviewed edit**, and
match-seed registration, study execution and the push remain unauthorized.

This establishes a population. It establishes **nothing** about strength — and
the population is one nobody plays, by the claim the artifact carries.
