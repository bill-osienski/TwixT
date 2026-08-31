# Low-Ply T1j Qualification — RAN ONCE, verdict `FAIL`

**A `FAIL` here is a RESULT, not an abort.** The run completed, spent its full
36-query budget, and recorded every reply. That is the whole reason this
qualification existed: D1 conflated FAIL with VOID and destroyed the observation
it was trying to make.

```
exit 0 · 2026-08-31T14:35:02Z → 14:35:09Z (7 s) · 36/36 queries · 7.0 s of 900 s
verdict FAIL · 29 recorded failures · record written
```

**Gates:** opened in its own one-line commit, restored to `False` immediately
after the run and before the suite was allowed to run again. All four now
`False`. Suite **3,628 passed / 4 skipped / 0 failed**. The thirteen
build-evidence files still verify against their manifest.

---

## 1. The finding

| ply | prefixes | depth 3 | depth 6 | `usealphabeta` | `currentMaxPly` | ms |
|---:|---:|---|---|---|---:|---:|
| **1** | 3 | **never completes** | **never completes** | `False` | 0 | ~25 |
| **3** | 3 | **never completes** | **never completes** | `False` | 0 | ~25 |
| **5** | 3 | completes | completes | `True` | depth+1 | 11 / 92 |

**12 of 12 invocations fail at ply 1. 12 of 12 fail at ply 3. 0 of 12 fail at
ply 5.** Every E3b binding succeeded and every postcondition surface was read,
so the instrument was sound throughout: this is a fact about T1j, not about the
harness.

**What the record shows.** At plies 1 and 3, T1j **never enters its alpha-beta
deepening loop at all** — `usealphabeta=False`, `currentMaxPly=0`,
`completed_depth=-1`, and it returns in ~25 ms. It still returns a *legal* move,
and the two independent JVMs frequently return **different** moves, which is
consistent with a non-search path rather than a search. At ply 5 it searches
normally at both depths.

### 🔴 What this establishes, stated at the width of the evidence

**Within these nine frozen `t1j_red` prefixes: plies 1 and 3 fail the frozen
completion condition, and ply 5 passes it.** That is the claim, and it is the
whole claim.

**It does NOT establish a global T1j threshold.** An earlier version of this card
said "the boundary lies between ply 3 and ply 5", which asserts a property of the
engine from nine positions in one colour arm at three plies. Three plies observed
is not a boundary located; it is three plies observed. The corrected wording is
above.

E4's lowest previously queried ply was 6, so ply 5 had not been observed before
this run — but "observed to complete in twelve invocations across three prefixes"
is not the same as "qualified", and this card does not claim the latter.

⚠ **`eval_regime` is `early_moveNr_lt_8` for all nine prefixes — including the
three that pass.** So that flag is *not* the discriminator, independently
confirming the earlier correction that the early-move regime is fine. What
distinguishes ply 5 from plies 1 and 3 is not recorded here, and this
qualification does not establish it.

⚠ **Scope.** Nine prefixes, three openings, one colour arm (`t1j_red`), plies
1/3/5 only. It says nothing about plies 2 and 4, which the frozen selection
never retains, and nothing about other arms.

## 2. 🔴 Does this explain the D1 VOID? Not established.

The obvious reading is that D1 died on a low-ply position. **Checking refuted
the simple version of that claim.**

**D1's first manifest row is at ply 5** — and ply 5 completes cleanly
(`usealphabeta=True`, POSTCOND `failures=0`). D1's first position would not have
failed.

The first non-completing position in D1's processing order is at **index 101 of
227**. So D1 would have had to clear 101 positions inside its 198 s — about
1.96 s each, covering a replay, a 400-simulation incumbent search and four T1j
queries — before reaching one.

| | |
|---|---|
| **Consistent** | D1 aborted at `depth 3 invocation 0`, which is what a ply-1 or ply-3 position does at depth 3, and E4Preflight exits 3 exactly when `req(completed, …)` fails |
| **Unproven** | whether D1 actually reached index 101 |

**And it cannot be settled retroactively**, because D1's VOID wrote no record and
its message did not name the position. That is precisely the gap the diagnostic
repair closed — for future runs only.

## 3. What this does not decide

It does not amend §12.1. It does not reserve a seed interval. It does not
authorize a D1 retry. Those remain separate decisions, and this result is one
input to the first of them.

## 4. Deviation from the card, recorded not hidden

The card froze the output file table as `02_prerun_verification.txt` …
`08_full_suite_after_gate_restored.txt`. Those numbers were consumed by the
**runner-build evidence** (02–13) committed between the card being frozen and the
run being authorized, and evidence is create-only. The run's files therefore take
the next free numbers with the card's semantic names, in the card's frozen
**location**:

`14_prerun_verification` · `15_run.sh` · `16_stdout` · `17_stderr` · `18_exit` ·
`19_lowply_records.json` · `20_full_suite_after_gate_restored` · `21_timing`

## 5. Run conditions, verified before launch

Frozen input hash-verified (`a9054cb2…`), toolchain resolved from the durable
root with 5 artifacts hashed, limits read from the code (36 / 120 s / 900 s), and
the runner confirmed to contain **no** reference to `SEED_INTERVAL`,
`seed_is_accounted`, `rng_stream_seeds`, `random.Random`,
`load_reference_evaluator` or `SeededReferenceAgent` — it never invokes the
incumbent, so it needed no seed interval and used none.
