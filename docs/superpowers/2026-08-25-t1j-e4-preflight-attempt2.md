# E4 preflight attempt 2 — corrective. RAN, PASSED. No games.

**Date:** 2026-08-25 · **Status:** attempt 2 **RAN and PASSED.** Driver gate exit **0** — unit tests
0, sweep 0, determinism 0, freeze 0, preference controls 0, preference pair 0.
**No game played, no seed consumed, no model loaded, no T1j source modified.**
**Supersedes [attempt 1](2026-08-25-t1j-e4-preflight.md)**, preserved unchanged and asserted
byte-for-byte by the driver before anything ran. · **The endpoint screen remains UNAUTHORIZED.**

Basis: `main` @ `4bb4639`. Evidence: `evidence/2026-08-25-t1j-e4-preflight-attempt2/` — 20 files
plus a **self-excluding manifest**. Full suite at this tree: **2913 passed, 4 skipped, 0 failed**
(`19_full_suite.txt`).

---

## The five findings, all real, all closed

### 1 — The selector used the wrong frozen metric

Attempt 1's preregistration already said *maximum per-query **wall** time*; its sweep populated the
statistic from the helper's internal `elapsed_us`. The gate did not compute the metric it had frozen.

**No threshold changed.** The selector now consumes wall ms; engine time is retained as a
descriptive field and both are persisted per query. A self-test control pins the fix: *a depth whose
engine time is tiny but whose wall time is over the ceiling must be rejected* — reading engine ms
would pass it.

| depth | **wall ms** (selector) | engine ms (descriptive) | verdict |
|---|---:|---:|---|
| 3 | 224 | 111 | qualifies |
| 4 | 306 | 191 | qualifies |
| 5 | 603 | 492 | qualifies |
| 6 | **2789** | 2670 | **selected** |
| 7 | 32193 | 32078 | rejected — over the frozen 30000 |

Same endpoints as attempt 1, now reached by the rule as written.

### 2 — The tasks contained no endpoint, and no opponent

Every task identity now carries **endpoint, `mdPly`, opening, colour arm, reference and reference
hash**, and the count is derived from all dimensions:

```
e4screen-000-strong6-o1_center-t1j_red
  endpoint=strong  t1j_mdPly=6  t1j_mdFixedPly=true  opening=o1_center
  colour_arm=t1j_red  reference=calib020_0001
  reference_sha256=34c79c0d…  seed=202612000
```

**8 openings × 2 colours × 2 endpoints = 32 games, 16 per endpoint** — asserted per endpoint for
balanced arms and complete opening coverage. 16 per endpoint is the smaller of the two options you
offered, taken at the endpoint level rather than by cutting to four openings, so each endpoint gets
a full balanced sub-screen.

### 3 — Our side is now a reproducible agent identity

| field | value |
|---|---|
| reference | `calib020_0001` |
| path | `checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors` |
| sha256 | `34c79c0d85a837f0281e90bb6a132c41535dc7830729fdd872af7df9612fcc26` |
| sha1 | `209cf2d4…` — **matches the value the twixtbot G3 preflight recorded**, independently confirming the same bytes |
| search | 400 sims, `eval_batch_size` 14, `stall_flush_sims` 48, **`dirichlet_eps` 0.0 — root noise OFF**, workers 1, `max_moves` 280 |
| defaults | snapshotted from `MCTSConfig` at freeze time, so the record cannot drift from the code |

Noise is off deliberately: a screen measures playing strength, and root noise would add variance the
seeds cannot remove. Temperature is kept, which is what makes the per-task seed load-bearing — T1j
at fixed ply is deterministic, so our side is the only stochastic party.

### 4 — The result classifier is now executable

**Draws are defined.** TwixT cannot draw by rule, so a draw is *always* a ply-cap termination:
`t1j_win` 1.0, `draw` 0.5, `t1j_loss` 0.0, `terminal_reason` recorded. If **more than half** the
games at an endpoint end by cap, that endpoint is `INCOMPLETE` — a score rate over unresolved games
means little.

**Scores are separated by endpoint.** Each endpoint gets its own decision over its own 16 games;
the joint outcome is a function of the pair, and the truth table is **total over all 4² = 16
combinations** and asserts so.

**`INCONCLUSIVE` has a numeric rule:** it is exactly the case where either endpoint is `INCOMPLETE`
— fewer than 16 resolved, or the cap-saturation abort fired.

**The early stop is now reachable, and its asymmetry is stated rather than hidden.** With band
`[0.05, 0.95]` and n = 16, a *saturated* verdict needs ≥ 15.2 or ≤ 0.8 points, so it **can never be
forced before game 16** — attempt 1's "saturation early-stop checked every 4 games" could never
fire, exactly as you said. What *can* fire early is the opposite: stop as soon as **both** saturation
verdicts are arithmetically unreachable and record `IN_BAND`. One win and one loss gives 1 point
after two games, whence `1 + 14 = 15 < 15.2` and `1 > 0.8` — **`IN_BAND` is forced after two games.**

| | weak | strong | joint |
|---|---|---|---|
| both saturate high | `SATURATED_STRONG` | `SATURATED_STRONG` | `T1J_TOO_STRONG` — stop |
| both saturate low | `SATURATED_WEAK` | `SATURATED_WEAK` | `T1J_TOO_WEAK` — stop |
| dial spans the band | `SATURATED_WEAK` | `SATURATED_STRONG` | `BRACKETED` — larger match at an intermediate depth |
| either in band | any | `IN_BAND` | `IN_BAND` — larger match at the in-band endpoint |
| either incomplete | any | `INCOMPLETE` | `INCONCLUSIVE` |

### 5 — Determinism records are persisted, bound, and structurally checked

All **25 records persisted individually** before any aggregate (25 `determinism_query` rows). The
**20 + 5 process structure is asserted from the helper's own `PROC` lines** — one pid across the 20
shared records, **five distinct pids** across the fresh ones, no overlap, shared query ids exactly
1..20 — rather than from how the harness believes it invoked them. And **every one of the 25
returned dumps is re-bound** through the same searched-versus-bound E3b comparison the sweep uses,
before the aggregate may pass.

```
process structure        1 shared pid [65528] + 5 distinct fresh pids [65530..65534]
distinct moves           ['(14, 11)']          distinct completed depth [6]
dumps re-bound clean     25/25                 14.4s of the 900s budget
```

## Runtime, recomputed without the withdrawn assumption

Attempt 1 asserted 60 T1j moves per game. **That assumption is withdrawn.** This preflight measured
single queries from 6–14 ply positions; it measured no game and no game length. The only defensible
bound is the ply cap: ≤ 280 plies ⇒ **≤ 140 T1j moves per game**.

| basis | weak (`mdPly` 3) | strong (`mdPly` 6) |
|---|---:|---:|
| **ceiling — ply-cap bound** | **8.4 min** | **104.1 min** |
| 20 moves/game — illustrative only | 1.2 min | 14.9 min |
| 60 moves/game — illustrative only | 3.6 min | 44.6 min |

Minutes of **T1j search time per endpoint**, 16 games each. The ceiling row is a bound; the other two
are illustrative and no measurement supports them. **Our side's time is excluded entirely** — it has
never been timed in this configuration. Whole-screen ceiling for T1j's side: **≈ 113 min**.

## Unchanged from attempt 1, re-measured

Dial response **`observable_move_response`** — 14 (position, depth) pairs differ from depth 3 across
5 of 6 positions; all 30 queries completed their requested depth. Cost still tracks **position
structure, not ply count**. Six positions, both evaluation regimes, both sides to move, each bound
through the E3b adapter at `ply_cap = 280` with abort on first divergence.

---

## What this establishes, and what it does not

**Established:** the endpoint selection now computes the metric it froze; the screen schedule is
executable — every task names its endpoint, opponent and settings, draws are defined, decisions are
per-endpoint, the joint classifier is total, and the early stop can actually fire; determinism at
`mdPly = 6` holds across 25 individually persisted, individually re-bound records with the process
structure asserted from the helper's own output.

**Not established, and not claimed:** strength monotonicity; that depth 6 is stronger than depth 3;
determinism beyond one position at one depth; **any game length whatsoever**; our side's runtime;
and absolute placement — the E0 caveat stands, so even a passing screen yields an **ordering**, not
a placement.
