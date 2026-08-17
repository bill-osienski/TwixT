# Corrections to FALSIFICATION.md

`56e1a83` recorded the lazy falsification. Review found three claims in that memo — and in its
commit message — that the measurement does not support. `56e1a83` is **not rewritten**; the
corrections are recorded here and applied to `FALSIFICATION.md`, so the original assertion and its
retraction both remain visible.

The **raw evidence is unaffected**: `stdout.txt` is byte-identical to what the run produced
(sha256 `fc497b32a2b8452c8711d3c38aa2636c39588d725cdce231f2c0a6885744099c`, 924 bytes), and the
measurement itself — exit 0, signal null, `copy_count 8`, gate satisfied — stands unchanged.

## 1. "only the execution surface differs" — FALSE

The falsification harness itself was revised between `481f9bd` and `3189187`. Verified by diff:

```
server/mcts.js                 78 +++++----
tests/mcts_golden/cases.mjs   137 +++++++++++++-----
tests/mcts_golden/falsify.mjs  68 +++-------
tests/mcts_golden/worker.mjs   26 +++++-----
```

Stage selection, the surface guards and the attribution plumbing all changed. Calling the pair a
one-variable comparison overstated it.

**What actually makes the two counts comparable**, each verified rather than assumed:

- `measureCopies` — the measurement window — is **byte-unchanged**;
- the frozen `FALSIFICATION` parameters (position, model, `S`, `cPuct`, gate) are
  **byte-unchanged**;
- `readFixture`'s body is unchanged; the worker diff is stage plumbing only;
- the fixture resolves identically: `P11` @ prefix 28, `n_legal` 500.

So this is **two measurements taken the same way**, not a single-variable experiment.

## 2. "none terminated early at an already-materialized node and none reached a terminal leaf" — WITHDRAWN

A copy count says nothing about traversal shape. Eight copies is consistent with both of the
excluded cases:

- a simulation may descend through **any number of already-materialized nodes** before
  materializing one new child at the leaf, and still contribute exactly one copy;
- a **newly materialized child may itself be terminal** — descent materializes it, then the loop
  exits and the terminal value is taken without expansion — which also costs exactly one copy.

What the count **does** support, combined with §3's at-most-one-per-simulation bound, is the
narrower statement retained in the memo: **every one of the eight simulations materialized exactly
one new child.**

## 3. A general bound stated as if measured — RECONCILED

The memo claimed "state construction is bounded by the simulation count rather than scaling with
the 500 legal moves" two lines above an explicit "no scaling law" disclaimer. The two could not
both hold.

The memo now reports the **observation** — 8 copies for 8 simulations at a position with 500 legal
moves — and leaves the general bound to the **structural argument in design §3**, which is where
it belongs and where it is argued rather than measured.

## Unaffected

The measurement, the raw stdout, the eager/lazy contrast (4490 vs 8), the completion of §5 in both
directions, and every "what this does NOT establish" statement in the memo are unchanged. No
execution-surface file was touched by `56e1a83` or by this correction.
