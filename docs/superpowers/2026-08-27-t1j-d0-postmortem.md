# D0 — Zero-Inference Postmortem of the Canonical 64-Game Match

**Status:** COMPLETE. Read-only. No model was loaded, no JVM started, no T1j
query issued, no inference run, no game played, no seed drawn, nothing trained,
nothing pushed.

**Authorization:** D0 only, granted 2026-08-27. No phase inherits authorization
from this one; D1 remains unauthorized.

**Bound to:** repository `d2862ab353f0` · record sha256
`42180c1f27406ad2…` · plan sha256 `c8b9cba816852a67…` ·
task digest `193d66bf5f1e4dca…` · 64 canonical
task identities · checkpoint `209cf2d4fd24…` · T1j jar
`53ec95e421db…` and all four JDK component hashes.

---

## 1. P0 — the record closed before analysis

| Check | Result |
|---|---|
| R1 says only what its bounded survey established | **VERIFIED, not re-amended** — `58f21d4` carries the correction; the only surviving mention of the universal SGF/π claim is the line withdrawing it |
| Canonical L0 match directory preserved byte-for-byte | **VERIFIED against `A_MANIFEST.sha256.txt`: 16 artifacts checked, 16 byte-identical, 0 mismatched, 0 missing.** The count is printed so a check that silently failed to run cannot read as a pass |
| Analysis bound to the JSONL and embedded plan by digest | **VERIFIED before any move was read** — `bind_record` is the sole producer of a `Bound`, and every function that can expose a move requires one |
| Repository commit, checkpoint / JAR / JDK hashes, 64 task identities recorded | **`01_identity.json`** |
| Role change reconciled | T1j is a development sparring partner from L0 onward; the pre-development L0 result is untouched |

Digest binding is a *precondition*, not a peer step. Four negative controls
prove it refuses: an edited header digest, an edited embedded task, a
non-pinned plan file, and an attempt to read moves without a `Bound` at all.

## 2. Inventory — both halves, integrity and outcomes only

All **64** games reconstruct from the embedded opening prefix plus
the recorded plies: **64/64** replay to
their recorded winner *and* recorded length through our own rules engine.

| Half | Games | T1j points | Winners | Terminal reasons |
|---|---:|---:|---|---|
| Discovery (reps 0,1) | 32 | 17.0 | black 9, red 23 | win 32 |
| Confirmation (reps 2,3) | 32 | 21.0 | black 11, red 21 | win 32 |

Confirmation **outcome counts** are open by §4.2; confirmation **diagnostics**
were never computed. That is enforced by two separate functions — `inventory`
works on all 64, `game_features` refuses anything outside reps 0 and 1 — not by
a flag, because a flag has a default and a default is a switch-off. Negative
controls assert the refusal for rep 2 and rep 3, and a signature check asserts
`game_features` takes no override parameter.

## 3. Discovery-half diagnostics

**1,329 plies over 32 games.**
Every §4.3 feature was derived, aggregated by opening, colour arm, winner and
coarse game phase (`03_aggregates.json`). **Each column carries its own
denominator**, not the cell's row count: `min_goal*_distance_*` is undefined
before a colour has a peg, and averaging it over rows where it never existed
understated it — in the opening phase, over 160 plies rather than 192. Caught by
a test written for it, fixed before the aggregates were published.

Two derivations needed defining, as Amendment 1 required:

- **Per-component boundary distance** is new — no existing helper supplies it.
  Tensor channels 19–22 are per-cell geometric edge distance and
  `connectivity_masks` reports only goal contact. The metric implemented is
  **geometric**, chosen because a graph distance would smuggle in a judgement
  about how an intervening hole could be filled, which D0 may not make.
- **One-ply terminal-threat detection** was **cost-measured before use**:
  ~2.7 ms/ply on a 3-game sample, ~4 s projected for the half. Comfortably under
  any stop rule, so it is computed exhaustively with no pruning to go wrong.

## 4. The §4.5 gate

**Verdict as specified: `GO`**, from 3 candidates.

| Signature | Hits | Openings | Arms | Verdict | Reason |
|---|---:|---:|---:|---|---|
| `unanswered_immediate_threat` | 32 | 8 | 2 | **REJECTED** | every hit sits at the same distance [2] from the terminal ply, so the signature restates the ending rather than a structural pattern |
| `mover_fragmentation` | 476 | 8 | 2 | **PASSES** | &mdash; |
| `created_threat` | 67 | 8 | 2 | **PASSES** | &mdash; |

> ⚠ **DEVIATION FROM §4.5, FLAGGED FOR REVIEW.** The plan specifies **four**
> gate conditions. This implementation enforces **five**. The fifth is described
> below. It does not invent a new requirement: it mechanizes a prohibition §4.1
> already states in prose ("call a move 'bad' solely because the mover later
> lost"), and it makes the gate strictly more conservative. It was added *before*
> the by-system cut in §5 was computed. Contrast §6, where a sixth condition
> would have changed the verdict and was therefore **not** added.

`unanswered_immediate_threat` was rejected mechanically, and the rejection is
the most instructive result in this report. All **32** of its hits are the
penultimate ply of the **32** games. It is a restatement of "every game ended in
a win", not a structural pattern — precisely the "call a move bad because the
mover later lost" failure §4.1 forbids. The gate as originally written would
have passed it on all four conditions; a fifth condition was added, with a
failing test first, to catch a signature whose every hit sits at one fixed
distance from the terminal ply.

## 5. 🔴 What the gate does not test — and it decides the answer

§4.5 asks whether a signature is *well-formed*: recurring across openings, present
in both arms, recomputable on the holdout, mapped to a D1 observable. It does
**not** ask whether the signature distinguishes `calib020_0001` from T1j. Cut by
which engine actually moved (`05_by_system.json`):

| Signature | Our rate | T1j rate | Games where ours exceeds T1j |
|---|---|---|---:|
| `created_threat` | 33/664 = **0.050** | 34/665 = **0.051** | 15/32 |
| `ignored_threat` | 17/664 = **0.026** | 15/665 = **0.023** | 17/32 |
| `immediate_win` | 15/664 = **0.023** | 17/665 = **0.026** | 15/32 |
| `mover_more_fragmented` | 219/664 = **0.330** | 257/665 = **0.386** | 19/32 |

**No candidate is asymmetric against us.** `created_threat` is
0.050 versus
0.051 — indistinguishable.
`mover_more_fragmented` runs
0.330 for us against
0.386 for T1j: the
signature recurs, but *we are the less fragmented side*. The direction even
disagrees between denominators — at ply level T1j leads, while our per-game
count exceeds T1j's in
19/32
games. Plies within one game are not independent, so the ply rate overstates its
own precision; both denominators are reported rather than the flattering one.

The late-game fragmentation split that looks dramatic — the eventual loser holds
roughly twice the component count of the eventual winner — holds at nearly
identical magnitude for **both** engines. It is a structural property of losing
at TwixT, not a property of our model.

## 6. What D0 concludes, and what it does not

D0 may say a structural signature **recurs**. It does:
`mover_fragmentation` and `created_threat` both recur across all 8 openings and
both colour arms, and both are recomputable on the held-out half.

D0 may **not** assign any of this to policy, value or search — those observables
were never captured — and it does not. No move here is labelled good or bad.

**The honest summary is a split verdict:** the gate as written returns `GO`,
while the substance is that no measured signature identifies a weakness of
`calib020_0001`. §4.5's "otherwise NO_GO" clause describes the alternative as
"no *actionable* repeated weakness", and by that word this result is a NO_GO.
The gate's four conditions do not contain the asymmetry test that word implies.

That gap is reported, not patched. Tightening the gate after seeing which
signatures survive is exactly the "only survives by changing the cohort, metric
or interpretation" move §6 rules out, and it would be no more legitimate applied
in the restrictive direction. Whether to amend §4.5 to require engine asymmetry,
or to proceed to D1 with signatures known to be symmetric, is a decision for
review — not one D0 may take for itself.

## 7. Evidence

| File | Contents |
|---|---|
| `01_identity.json` | repository, record, plan, task digests, checkpoint/JAR/JDK hashes, 64 task ids |
| `02_inventory.json` | both halves: integrity, reconstruction, outcome counts |
| `03_aggregates.json` | discovery-half aggregates by opening, colour arm, winner, phase |
| `04_gate.json` | the §4.5 gate, its candidates and per-condition failures |
| `05_by_system.json` | each signature split by engine, at both denominators |
| `06_full_suite.txt` | full repository suite after the change |

Reproduce with:

```
.venv/bin/python -m scripts.GPU.alphazero.d0_postmortem \
  --record docs/superpowers/evidence/2026-08-27-t1j-l0-canonical-match/06_l0_match_results.jsonl \
  --plan   docs/superpowers/evidence/2026-08-26-t1j-l0-larger-match/01_l0_match_plan.json \
  --out    <fresh directory>
```

The command is create-only and refuses to overwrite an existing artifact.
