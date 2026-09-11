# H2 SEED PREPARATION — `[202618000, 202618736)` registered as ACCOUNTED only

**Authorized scope:** re-prove the interval, register it in `ACCOUNTED_SEED_INTERVALS` only, verify
736 accounted / 0 exposed / 0 retired with all seven gates still False, package and commit.

**Not done, not authorized:** the H2 gate, Java, the model, gameplay, any change to D1, training,
the push. Nothing was executed and no seed was drawn.

## 1. The re-proof

Against the registries **as they now stand** — which since the D1 round gained D1's own §14 block,
spent and retired — plus this paper reservation itself and every derived RNG stream.

| | |
|---|---|
| prior seeds | 4,845 across six categories |
| prior values incl. derivations | 24,225 |
| direct overlaps | **0** |
| derived-stream collisions | **0** |
| own derivations | 3,680 = 736 × 5, injective |
| enumeration | EXHAUSTIVE (widest interval 800), not sampled |
| collision controls | **13 of 13 rejected** |
| gap-policy controls | **4 of 4 rejected** |

🔑 **The candidate is H2's own paper reservation**, so it is excluded from the prior set **by
identity** — in the **gap** check as well as the overlap check, which is the correction the D1 round
had to make mid-proof and which this one inherits rather than rediscovers.

🔑 **The gap floor is now the candidate's own size (736, not the fixed 224).** A gap smaller than the
block it protects is not a load-bearing gap. Nearest other boundary: **776**.

**D1's block is a control now, not a paper reservation.** It was drawn and retired whole on
2026-09-08, so it lives in the registries and is rejected through them; listing it again as paper
would have been double-counting dressed as extra safety.

## 2. The registration

One tuple added to `ACCOUNTED_SEED_INTERVALS`. The registry file's diff is a **pure insertion** —
**zero lines removed**. `EXPOSED`, `RETIRED`, `TEST_ONLY` and `CONSUMED_SEEDS` are untouched, and
D1's block still reads 221 / 221 / 221.

Per seed, all 736: **accounted 736, exposed 0, retired 0, test-only 0, consumed 0.**

## 3. The barrier is down; the gate is not

| | |
|---|---|
| `check_seed_registration()` | **SATISFIED** |
| all seven gates, as imported | **False** (`is False`, type `bool`) |
| `run_h2(...)` | still raises: execution UNAUTHORIZED |

A control that flips `H2_EXECUTION_AUTHORIZED` to True is rejected by the same test that asserts the
barrier, so "the gate stayed shut" is a claim a test can contradict.

## 4. Verification

| | |
|---|---|
| injected-defect controls | **520 / 520 rejected**, 0 not caught, 0 stale, 0 duplicates |
| clean baseline | **PASS** over 427 target tests |
| every source restored | True |
| full suite | **4,385 passed, 4 skipped, 0 failed** |

Five controls are new: un-registering the block, marking it EXPOSED, marking it RETIRED, opening the
gate, and a negative control that stops stripping.

## 5. Two faults this round, both caught by the machinery

1. 🔴 **A test elsewhere rotted the moment I registered the block.** `test_h1_viability` used the
   literal `202618000` as "a seed in no registry at all" — which became **H2's first seed**. It now
   **searches** for an unregistered seed and asserts it found one, so it cannot rot again on the next
   block. The full suite caught this, not me.
2. **A control could not see its defect.** "The registration barrier is disabled" pointed at the
   inverted test, which asserts the barrier is *satisfied* — something a disabled barrier also
   satisfies. Re-aimed at the test that strips the block.

Both are the same shape as the D1 round's, and the inherited fix — assert the strip bit, aim the
control at the test that strips — is what made the second one visible.

## 6. What is next, and what is not

The **736-game match is a separate authorization**. Nothing here opens the gate, and the block is
accounted but never drawn: a reservation is not a draw.
