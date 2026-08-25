# E4 harness qualification attempt 3 — corrective. RAN, PASSED. No model loaded.

**Date:** 2026-08-25 · **Status:** **RAN and PASSED.** Driver gate exit **0**.
**No model was loaded, no T1j, no game, no scheduled seed.**
**Supersedes [attempt 2](2026-08-25-t1j-e4-harness-qualification-attempt2.md)**; attempts 1 and 2
preserved unchanged. · **The 32-game E4 screen remains UNAUTHORIZED.**

Basis: `main` @ `0a441eb`. Evidence: 8 files plus a self-excluding manifest.
Full suite: **3015 passed, 4 skipped, 0 failed**. 102 harness tests.

Attempt 2's single real reference-agent call **stands and is not repeated** — all four defects
corrected here live in the harness, and every one is exercisable with a tagged fake evaluator.

---

## 1 — [P1] Zero games produced a permissive verdict

The public `qualify` path classified **zero results** and emitted `joint=INCONCLUSIVE` with
`larger_match_permitted=true`, exit 0. That is a screen conclusion drawn from no games.

A screen verdict now requires a **structurally complete result set**: every scheduled task either
produced a result or was skipped by a **recorded early stop for its own endpoint**. Otherwise the
run emits a **`qualification_receipt`** carrying no `joint` and no `larger_match_permitted`.

```
ZERO GAMES PRODUCE NO SCREEN VERDICT
withheld because: no tasks were scheduled; a screen verdict needs a screen
```

| result set | verdict? |
|---|---|
| zero tasks | **no** |
| a missing result | **no** |
| a duplicated result | **no** |
| an alien identity | **no** |
| an **unjustified** skip | **no** |
| a skip justified by a recorded early stop | yes |
| a complete set | yes |

## 2 — [P1] T1j would have failed the evaluator identity gate

The gate applied to **both** colours, so the first T1j construction would have aborted — T1j is
classical and holds no MLX evaluator. It now applies **only when `mover == task["reference_colour"]`**.

```
PASS  a CLASSICAL anchor agent with NO evaluator is accepted alongside the reference
PASS  NEGATIVE CONTROL: the REFERENCE side rejects no evaluator at all
PASS  NEGATIVE CONTROL: the REFERENCE side rejects a rebuilt evaluator
PASS  ...and the SAME agent is accepted on the anchor side
```

Both a *missing* and a *rebuilt* evaluator are rejected on the reference side.

## 3 — [P1] The opening was never bound

The scripted six-ply opening is a position both engines must already agree on, yet the first binder
call came only after ply 7 — a divergent opening could have run a whole game unnoticed. It is now
bound **before the terminal check and before either agent is constructed**, and the bind is recorded.

```
PASS  the opening was bound first, at ply 0
PASS  then every applied move: [0, 1, 2, 3, 4, 5, 6, 7]
PASS  NEGATIVE CONTROL: an opening divergence aborts
PASS  ...and NO agent was constructed before the opening was bound
```

## 4 — [P1] Binder failures escaped unclassified

Only `AbortError` was handled, so a plain `ValueError` propagated as *unexpected* and left no durable
abort record. Every binder call now goes through `_bind`, which wraps anything else as
`AbortError(PHASE_BIND, …)`. And **cleanup now runs once for every *started* task**, including one
that aborted — with the original failure preserved if cleanup fails too.

```
PASS  a PLAIN ValueError from the binder is classified as PHASE_BIND
PASS  ...and a durable abort record was written
PASS  cleanup ran for the STARTED task even though it aborted
PASS  a cleanup failure does NOT mask the abort that preceded it
PASS  ...and the cleanup failure is recorded separately
```

---

## What this establishes, and what it does not

**Established:** a screen verdict cannot be drawn from an incomplete result set; the identity gate
binds the side that actually loads a network and admits a classical opponent; the opening is bound
before play begins; and no binder failure escapes classification or loses its abort record.

**Not established, and not claimed:** anything new about the model — **attempt 3 loaded none**, and
the one real call remains attempt 2's; anything about T1j — **the per-ply binder is still a refusing
stub**, so nothing binds two engines yet; that the agent plays well; game length or screen runtime;
and absolute placement.
