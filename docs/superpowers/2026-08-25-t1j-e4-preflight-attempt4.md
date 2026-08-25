# E4 preflight attempt 4 — corrective. RAN, PASSED. No games.

**Date:** 2026-08-25 · **Status:** attempt 4 **RAN and PASSED.** Driver gate exit **0**.
**No game played, NO MODEL LOADED, no T1j source modified. 32 scheduled seeds were spent by
attempt 3 and are recorded below.**
**Supersedes [attempt 3](2026-08-25-t1j-e4-preflight-attempt3.md)**; attempts 1–3 preserved
unchanged and asserted byte-for-byte by the driver. · **The endpoint screen remains UNAUTHORIZED.**

Basis: `main` @ `73b4119`. Evidence: `evidence/2026-08-25-t1j-e4-preflight-attempt4/` — 20 files
plus a **self-excluding manifest**. Full suite: **2932 passed, 4 skipped, 0 failed**.

---

## 1 — [P1] Attempt 3 spent 32 scheduled seeds. Recorded, not glossed.

`rng_witness` constructs **both** derived generators and **draws four values from each**. Attempt 3
called it on every scheduled task. No model was loaded and no game was played, but under this
workstream's accounting **`[202612000, 202612032)` is spent**.

| | |
|---|---|
| burnt interval | **`[202612000, 202612032)`** — 32 seeds |
| spent by | E4 preflight attempt 3, commit `73b4119` |
| how | `rng_witness` per task: two generators constructed, four draws each |
| model loaded / game played | **no / no** |
| status | **EXPOSED** — registered in `EXPOSED_SEED_INTERVALS`, refused by `validate_task` |
| replacement | **`[202612128, 202612160)`**, past the burnt prefix with a gap |

The gap-and-move follows the twixtbot G3 precedent, which burnt `202611000` and re-froze to
`[202611128, 202611256)`.

**The schedule no longer draws.** Tasks carry `rng_streams` — the **XOR-derived stream integers
only**, since deriving an integer is not a draw and spends nothing. And the mistake is now
fail-closed in both directions:

```
rng_witness(scheduled seed)  → E4ReferenceError: refusing to draw from scheduled
                                seed 202612128: drawing spends it.
validate_task(burnt seed)    → E4ReferenceError: seed 202612000 was EXPOSED by an
                                earlier witness and cannot be scheduled
```

The witness **mechanism** is still evidenced — demonstrated once on synthetic seed `90000001`,
outside every accounted interval.

## 2 — [P1] The construction path is now exercised

Attempt 3's tests compared a separately computed witness and **never called `build()`**. A
regression that stopped delegating, changed the task contract, or seeded different generators would
have left every test green.

A no-model test now builds through the real path with a fake evaluator carrying the identity tags
`load_reference_evaluator` sets, and compares **both generator states**:

```python
agent = E.build(task(seed=SYNTHETIC, anchor="red"), evaluator=_FakeEvaluator())
assert agent.mcts.rng.getstate()    == random.Random(SYNTHETIC ^ SEARCH_MASK["black"]).getstate()
assert agent.readout_rng.getstate() == random.Random(SYNTHETIC ^ READOUT_MASK["black"]).getstate()
```

Plus: `build` refuses an evaluator from a different checkpoint, refuses one with no identity tags at
all, refuses a task missing a required field, and produces different search states for the two
colours. **19 tests in this file; no model is loaded and no move generated.**

## 3 — [P2] The binder now compares the blob column

Attempt 3 parsed a git blob id per row and **never compared it**, so its evidence claimed a
committed-source binding while checking only sha256 against the worktree. Every tracked path is now
bound on **four** facts — transcript digest, manifest sha256, disk sha256, and the **git object
store** — with the object id read from the index, which is the blob the pending commit holds.

**The checkpoint is handled separately and labelled.** `checkpoints/` is gitignored, so
`model_iter_0001.safetensors` has **no object in any commit** and `git rev-parse <commit>:<path>`
cannot succeed for it. Its row reads *manifest+disk ONLY — UNTRACKED*.

```
BINDING OK — 9 tracked paths bound to the object store, 1 untracked reported as such
```

Nine controls, all invoking the real checker, all passing — including **manifest blob id flipped**,
**object store disagrees with the manifest**, **a tracked file reported as untracked**, and an
acceptance control.

> **A control caught my own checker again.** The first rewrite dropped the expected-path set, so
> *dropping a manifest row* made that path invisible rather than failing — the control reported
> `accepted (0)`. The set is restored and the control now rejects. That is the second time this
> workstream that a control, not a review, found the hole; it is why they run.

## Re-measured, unchanged in substance

Wall ms by depth: 3→213, 4→330, 5→588, **6→2735 selected**, 7→30749 **rejected** over the frozen
30000. Dial `observable_move_response`, all 30 queries completing their requested depth. Determinism
at `mdPly = 6`: 25/25 identical `(14,11)`, 1 shared pid + 5 distinct fresh pids, 25/25 dumps
re-bound. Schedule accepted: 32 tasks, 32 distinct seeds, 32 distinct stream pairs, streams disjoint.
Earliest early stop **game 8**, computed. Runtime ceiling: weak **8.1 min**, strong **102.1 min**.

---

## What this establishes, and what it does not

**Established:** the seed accounting is now fail-closed in both directions and the 32 burnt seeds are
on the record; the reference construction is exercised through its real entry point without a model;
and the evidence binder compares every fact it records, distinguishing tracked from untracked.

**Not established, and not claimed:** that the reference agent **plays** correctly — no model has
ever been loaded and no move generated; strength monotonicity; that depth 6 is stronger than depth 3;
determinism beyond one position at one depth; any game length; our side's runtime; and absolute
placement — the E0 caveat stands.
