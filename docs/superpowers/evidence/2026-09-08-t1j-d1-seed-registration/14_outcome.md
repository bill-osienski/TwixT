# D1 SEED PREPARATION -- outcome

**Authorized scope:** re-prove `[202615000, 202615221)`, register it as ACCOUNTED **only**, leave
every other registry unchanged, verify the registration barrier is satisfied **while all six gates
stay False**, run controls + suite, package evidence, commit locally.

**Not done, not authorized:** the D1 gate, model load, Java, queries, analysis of a real record,
training, push. Nothing was executed and no seed was drawn.

## 1. The re-proof

Against the registries **as they now stand** -- which since the last D1-side proof gained **both**
spent H1 blocks -- plus this paper reservation itself and every derived RNG stream.

| | |
|---|---|
| prior seeds | 4,109 across six categories |
| prior values incl. derivations | 20,545 |
| direct overlaps | **0** |
| derived-stream collisions | **0** |
| own derivations | 1,105 = 221 x 5, injective |
| enumeration | EXHAUSTIVE (widest interval 800), not sampled |
| collision controls | **11 of 11 rejected** |
| gap-policy controls | **4 of 4 rejected** |

🔑 The candidate **is** D1's own paper reservation, so it is excluded from the prior set **by
identity** -- not by subtracting its seeds, which would excuse any control equal to an existing
reservation.

🔴 **A fault of mine, and its correction.** The v5 run applied that identity exclusion to the overlap
check but **not** to the gap policy, so it measured the candidate's distance to **itself** and
reported `gap policy FAIL` at distance 0. v6 applies the exclusion in both places and adds a control
that reruns the candidate **without** it, so the artefact itself is now a control. Nearest **other**
boundary: **773 seeds** (D1's retired block's end), against a policy floor of 224. Both runs are
kept: `01`/`02` are v5, `03`/`04` are v6.

## 2. The registration

One tuple added to `ACCOUNTED_SEED_INTERVALS`. The diff of that file is a **pure insertion** -- no
line removed. `EXPOSED`, `RETIRED`, `TEST_ONLY` and `CONSUMED_SEEDS` are byte-identical.

Per seed, all 221: **accounted 221 / exposed 0 / retired 0 / test_only 0 / consumed 0.**

## 3. The barrier is down; the gate is not

| | |
|---|---|
| `_check_seed_registration()` | **SATISFIED** |
| all six gates, as imported | **False** (`is False`, type `bool`) |
| `run_d1(...)` in-process | still raises `D1Error`: execution UNAUTHORIZED |
| the CLI, **fresh subprocess** | **exit 5**, nothing written, `--out` path absent afterwards |

Registration is bookkeeping, not permission. A control that flips `D1_EXECUTION_AUTHORIZED` to True
is rejected by the same test that asserts the barrier, so "the gate stayed shut" is a claim a test
can contradict.

## 4. Controls and suite

| | |
|---|---|
| injected-defect controls | **381 / 381 rejected**, 0 not caught, 0 stale |
| clean baseline | **PASS** over 305 distinct target tests |
| every source restored | True |
| full suite | **4,153 passed, 4 skipped, 0 failed** (`.venv/bin/python`, 14m52s) |

Eight controls are new or re-aimed this round: un-registering the block (caught at the barrier **and**
at the state assertion), marking it EXPOSED, marking it RETIRED, opening the gate, a barrier that
computes nothing, the same block marked EXPOSED from the H1 side, and an H1 block moved to overlap it.

🔴 **Two failed clean baselines before this one, both mine, both real.**

1. **An orphaned control.** Renaming `test_the_REAL_registry_still_refuses_the_new_block` left a
   control naming a node id that no longer existed. pytest exits nonzero for a missing node, so it
   would have scored REJECTED for free. The baseline check caught it. Re-aimed at a test that strips
   the block, since with the block registered the defect is unobservable at the real registry.
2. **A test I missed in my own audit.** `test_D1s_reservation_stays_PAPER_ONLY` in the H1 file
   asserted the block was in **no** registry. My audit grep excluded H1 files, so I did not find it;
   the baseline did. It is inverted to assert what is still true -- accounted only, disjoint from both
   H1 blocks -- and the control that used to inject "D1's reservation is registered too" is gone,
   because that edit is now the legitimate state.

The harness also printed `FAIL` with no reason under it, because it scanned only stdout and pytest
reports a missing node id on stderr. It now scans both.

## 5. What this does and does not establish

It establishes that the block is reserved, disjoint, registered in exactly one list, and that
execution is still refused at both the API and the CLI. It establishes nothing about D1's outcome:
no position has been queried and no seed drawn. The single D1 acquisition is a separate
authorization.
