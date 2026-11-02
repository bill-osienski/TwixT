# Combination policy second cycle: Python pin and a new generation attempt (amendment, 2026-11-02)

**Status:** the user chose this pin and approved building the new attempt (2026-11-02). The text is pinned by sha256 in
`combination_policy_pilot.py` and returned with the build for review. It authorizes no seed procedure, real model loading,
gate opening, generation, training, Build B, evaluation, deployment or push.

This amends `docs/superpowers/2026-11-02-combination-policy-second-cycle-amendment.md`, which is also still pinned. Whatever
this amendment does not change still applies. The card, the first cycle's records and the first attempt's VOID record stay
as written.

## Why

The one approved `run cycle2_generate` went VOID at its first check (ledger entry 47). The interpreter is now Python 3.14.10,
and the module pins 3.14.9. No checkpoint was read and no game started. Its block is closed, and that kind stays one-shot.

## The pin

The second cycle's live kinds, `cycle2_generate2` and `cycle2_train`, require exactly:

| Package | Version |
|---|---|
| Python | 3.14.10 |
| MLX | 0.30.3 |
| NumPy | 2.4.1 |

Anything else is still a VOID. The pin is per kind: `VERSIONS` stays 3.14.9 for the first cycle's kinds and for
`cycle2_generate`. Build B's kinds would need to name this pin when they are built.

**What this does not claim.** Python 3.14.9 and 3.14.10 have not been compared on this code, so they are not claimed to behave
identically. Nothing here depends on that: the second cycle's games, fit and evaluation would all run on one interpreter, and
no first-cycle number is pooled with them. The first head was trained on 3.14.9, and its bytes and digests stay the binding.
The CPU timing record, taken on 3.14.9, remains an estimate for pricing the fit, with no new timing run.

**A check added to the process.** The live versions are read against this pin before the seed sequence and again before the
launch. A mismatch stops the work and returns to the user, so it cannot use up another block. The focused tests also compare
the live versions with this pin, because their seams otherwise supply the expected values.

## The new attempt

`run cycle2_generate2` is the same run on its own paths:

- the games, teacher, binding checks, teacher-reach probe, workers, batch cap, rows, held-out rule and admission of
  `cycle2_generate`;
- the ceiling: TERM 21,590 s, KILL 21,600 s, cleanup observation through 21,610 s;
- a fresh 1,000-seed block by the existing method and gap floor, with survey, pre-proof, registration, post-proof and closing;
- its own closed gate, `COMBINATION_POLICY_CYCLE2_GENERATE2_AUTHORIZED`;
- evidence at `docs/superpowers/evidence/combination-policy-cycle2-generate2`, with `-run`, `-seed-survey` and `-seed-closing`;
- rows at `~/twixt-cache/combination-policy-pilot/cycle2_generate2/`.

`run cycle2_train` now reads only this attempt's corpus, record and `outer.json`. Its ceiling stays a placeholder, re-derived
from this corpus's actual counts before any training approval.

An interrupted or timed-out run is incomplete and receives no score, and there is no automatic retry. The first attempt's seeds
stay exposed and retired, and that attempt admits nothing.
