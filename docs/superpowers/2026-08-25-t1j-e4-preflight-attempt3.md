# E4 preflight attempt 3 — corrective. RAN, PASSED. No games.

**Date:** 2026-08-25 · **Status:** attempt 3 **RAN and PASSED.** Driver gate exit **0**.
**No game played, no seed consumed, no model loaded, no T1j source modified.**
**Supersedes [attempt 2](2026-08-25-t1j-e4-preflight-attempt2.md)**; attempts
[1](2026-08-25-t1j-e4-preflight.md) and 2 are preserved unchanged and asserted byte-for-byte by the
driver before anything ran. · **The endpoint screen remains UNAUTHORIZED.**

Basis: `main` @ `9370225`. Evidence: `evidence/2026-08-25-t1j-e4-preflight-attempt3/` — 20 files
plus a **self-excluding manifest**. Full suite at this tree: **2924 passed, 4 skipped, 0 failed**.

---

## 1 — [P1] Early `IN_BAND` was not forced. The claim is withdrawn.

Attempt 2 ruled out **score** saturation and stopped there. But an endpoint can also end
`INCOMPLETE` through **ply-cap terminations**, and at game 2 that route is wide open: nine of the
fourteen unplayed games could still hit the cap and fire the cap-saturation abort.

**Attempt 2's claim that `IN_BAND` is forced after two games is wrong and is withdrawn.**

The rule now closes both routes:

```
early_in_band_forced  ⟺  score saturation unreachable
                     AND  cap_terminations + (n − played) ≤ n // 2
```

and the earliest reachable stop is **computed, not asserted** — `earliest_early_stop(16, [0.05,0.95])`
returns **game 8**, with zero cap terminations and a score in `[1.0, 7.0]`. Stating that number
wrongly is the exact defect the function exists to prevent, so it is derived every run.

New controls, all passing:

| control | result |
|---|---|
| 1 win + 1 loss at game 2 | **INCOMPLETE** — cap route still open |
| 4–4 at game 8, zero caps | `IN_BAND` — early stop fires |
| **negative:** one cap termination at game 8 | early stop **refused** |
| **negative:** 7.5 of 8 at game 8 | refused — strong saturation still reachable |
| a completed endpoint | not an *early* stop |
| earliest possible stop | **game 8**, not game 2 |

## 2 — [P1] The seed is now bound to the moves

Attempt 2 recorded a seed and froze no path from it to a move. Worse, its tasks lacked
`reference_sha1` and `anchor_colour`, so the qualified builder would have **refused every one of
them at construction time**.

**`scripts/GPU/alphazero/e4_screen_reference.py`** is committed and delegates to
`twixtbot_g3_reference.build_reference_agent` — the construction already qualified for the twixtbot
G3 calibration. It is not reimplemented. Every task now carries what that builder reads:

```json
{"task_id": "e4screen-000-strong6-o1_center-t1j_red",
 "endpoint": "strong", "t1j_mdPly": 6, "opening": "o1_center",
 "colour_arm": "t1j_red", "anchor_colour": "red", "reference_colour": "black",
 "reference": "calib020_0001", "reference_sha1": "209cf2d4…", "seed": 202612000,
 "rng": {"colour": "black", "search_seed": 206161786, "readout_seed": 204448028,
         "search_first": [0.3247…], "readout_first": [0.5721…]}}
```

The two streams use the **four colour-specific XOR masks imported from `SeededReferenceAgent`**,
never re-typed. Freezing both derived stream seeds and their first draws makes the seed→RNG binding
checkable later **without replaying a game or loading a network**.

`validate_schedule` accepted the whole schedule: **32 tasks, 32 distinct seeds, 32 distinct
(search, readout) stream pairs, search and readout streams disjoint** — two tasks sharing a stream
would silently correlate games the schedule presents as independent.

**The `dirichlet_eps: 0.0` override is withdrawn as cosmetic.** `eval_runner.cfg_from` constructs
`MCTSConfig` **without passing `dirichlet_eps`**, so the field keeps its default 0.25; root noise is
suppressed at the call site by `search_with_root(state, add_noise=False)`. A test checks this
against the code rather than restating it:

```python
cfg = cfg_from(eval_config())
assert cfg.dirichlet_eps == MCTSConfig().dirichlet_eps != 0.0
```

`our_settings` is now **read from the qualified path** via `frozen_settings()` instead of
transcribed, so it cannot drift.

## 3 — [P2] The report table contradicted the classifier

The card said *either endpoint `IN_BAND` wins*. The executable classifier gives `INCOMPLETE`
priority, and the preregistration agrees with the code. The row is corrected:

| | joint |
|---|---|
| both saturate high | `T1J_TOO_STRONG` — stop |
| both saturate low | `T1J_TOO_WEAK` — stop |
| one low, one high | `BRACKETED` — larger match at an intermediate depth |
| **either in band *and neither incomplete*** | `IN_BAND` |
| **either incomplete** | `INCONCLUSIVE` — takes priority over everything above |

## Re-measured, unchanged in substance

Wall ms by depth: 3→215, 4→339, 5→583, **6→2845 selected**, 7→31073 **rejected** over the frozen
30000 ceiling. Dial response `observable_move_response`; all 30 queries completed their requested
depth. Determinism at `mdPly = 6`: **25/25 identical `(14,11)`**, 1 shared pid + 5 distinct fresh
pids, **25/25 dumps re-bound**, 14.9 s of the 900 s budget.

Runtime ceiling (ply-cap bound, T1j search only, 16 games per endpoint): **weak 8.0 min, strong
106.2 min**, whole screen ≈ 114 min. Our side's time is still excluded — it has never been timed.

---

## What this establishes, and what it does not

**Established:** the early stop closes both routes out of `IN_BAND` and its earliest firing game is
derived rather than asserted; the schedule is accepted by the reference construction that would
actually build our side, with every task carrying the fields that builder reads and its seed bound
to both generator streams; and the report now agrees with the executable classifier.

**Not established, and not claimed:** that the reference agent *plays* correctly here — **no model
was loaded and no move was generated**; strength monotonicity; that depth 6 is stronger than depth 3;
determinism beyond one position at one depth; any game length; our side's runtime; and absolute
placement — the E0 caveat stands.
