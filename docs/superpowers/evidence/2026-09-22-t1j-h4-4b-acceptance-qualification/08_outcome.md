# H4 §4B ACCEPTANCE-MODE QUALIFICATION — RAN ONCE, verdict `CLEAN`

```
exit 0 · 2026-09-22T23:40:15 → 23:40:27 -0400 · 12 s wall (11.3 s on the run's clock) of 1800 s
72/72 subprocesses (56 agent calls + 16 binder calls) · 72 records · 72 distinct pids
16 task buckets · verdict CLEAN · matrix 3cc14ca9…d22f27cd · record 05_qualification.json
```

**Run exactly once, as authorized**, under the §4B card (`658ebe1`, amended
`525141a`, clarified `7d95363`) against the implementation reviewed at `bb046f3`.
Gate opened in its own one-line commit (`9cfcbc1`); one launch as a fresh
subprocess through the **production H4-mode agent and binder** and the **verified
default compiler**; the gate restored by the wrapper's EXIT trap to `bb046f3`,
byte-identical — read back from source: **14 gates, none open**
(`07_gate_after.txt`). No Java process survived.

`00_prerun_verification.txt` recomputed every pin before the gate opened, built
the production adapter with no subprocess (one H4 runtime, depth 6, query and
replay timeouts 120 s), confirmed the default compiler **is**
`d1_probe._default_compile`, ran a javac-only compile into a throwaway directory,
and ran the exact command with the gate still closed (exit 5, nothing created).

---

## 1. Which routine answered — by ply, through the production adapter

| ply | agent calls | classification | exit / failures |
|---:|---:|---|---|
| 0 | 5 | `native_initial_first` ×5 | 3 / 1 |
| 1 | 15 | `native_initial_second_to_fourth` ×15 | 3 / 1 |
| 2 | 15 | `native_initial_second_to_fourth` ×15 | 3 / 1 |
| 3 | 15 | `native_initial_second_to_fourth` ×15 | 3 / 1 |
| 4 | 3 | `searched` ×3 | 0 / 0 |
| 5 | 3 | `searched` ×3 | 0 / 0 |

🔑 **The production adapter ACCEPTED all 50 native replies** — the replies that
every one of the three default refusal sites (§2 of the card) rejects — and all 6
searched ones. It **refused nothing** it had to accept, and no search occurred at
plies 0–2.

⚠ **`native_initial_fifth_or_more` was NOT observed** — `fifthOrMoreMove()`
returned null at all six ply-4/5 positions. Its acceptance by the production path
rests on **constructed** tests only (`b86420d`, `bb046f3`). Not a stop.

## 2. The contracts — all held (re-checked from the record's fields)

| contract | observed |
|---|---|
| every record `accepted` | 72/72 |
| exactly one `PROC` per jvm | 72 records, **72 distinct pids** |
| query reflection 4 · replay reflection 1 | 56/56 · 16/16 |
| `MatchData` = `pieRule=false 24×24 ystarts=true identity=true` | 56/56 |
| native = exit 3 + failures 1 · searched = exit 0 + failures 0 | 50/50 · 6/6 |
| QUERY line: `q=1`, `moveNr` = ply, requested depth 6 | 56/56 |
| returned move legal in **our** engine (recomputed) | 56/56 |
| clean safety surface, field by field | 72/72 |
| replay exit 0, failures 0 | 16/16 |
| helper sources == the committed files | True |

⚠ **SCOPE OF THE RE-CHECK.** Unlike the repair run, this record keeps the
adapter's **structured** per-call fields, not raw stdout. Dump coherence (the
query JVM's reconstructed position, and each replay's final state) was checked by
the adapter's own `compare_state` during the run — every record is `accepted`,
which it could only be if that check passed — but it **cannot be re-derived from
the record** afterwards.

## 3. Realized moves — REPORTED, NOT JUDGED

T1j coordinates `(x, y)`; five fresh JVMs per randomized cell, one per searched:

| cell | realized |
|---|---|
| empty_board ply 0 | (13,13) (11,9) (13,9) (9,12) (11,9) |
| o1_center ply 1 · 2 · 3 | (11,16) (11,18) (11,18) (11,16) (11,18) · (7,12) (5,12) (7,12) (6,12) (5,12) · (12,5) (12,6) (12,5) (12,5) (12,6) |
| o3_low ply 1 · 2 · 3 | (11,6) (11,6) (11,8) (11,6) (11,6) · (19,12) (19,12) (17,12) (18,12) (18,12) · (10,7) (10,5) (10,6) (10,6) (10,7) |
| o4_high ply 1 · 2 · 3 | (12,15) (12,15) (12,17) (12,15) (12,16) · (16,11) (17,11) (18,11) (16,11) (17,11) · (13,18) (13,17) (13,17) (13,17) (13,16) |
| searched, plies 4 · 5 | o1_center (9,16) · (11,9); o3_low (11,10) · (11,7); o4_high (12,12) · (12,16) |

All five ply-0 moves lie in `x, y ∈ 9..14`, consistent with the no-pie branch.
No diversity threshold, no probability estimate.

## 4. Scope

**Establishes**, for the 16 frozen positions at depth 6, under the fresh-JVM
lifecycle: the **production** `T1jAgent` (via `make_agent_factory`) and binder
(via `make_binder`), on one H4-mode runtime, accepted every reply, refused none,
named the answering routine, and recorded one `PROC` per JVM in 16 preserved
task buckets.

**Does NOT establish**: real `fifthOrMoreMove()` output being accepted; anything
outside these positions or at another depth; the game loop, the H4 runner, the
pilot, or **strength**. No game, no seed, no score. Not poolable with the repair
qualification, §4A, H1, H2, H3 or L0 — the classification pattern matching the
repair run is an observation, not pooled evidence.

## 5. What this decides

§4B is **complete: CLEAN**. That is a precondition for the H4 runner/pilot
stage, **not an authorization**. Games, seeds, pilot execution, aggregation,
push, and any post-result implementation change remain unauthorized. Still owed
before any pilot (replacement card): the H4 runner, the §1.3 duplicate-pair
rule removal with its control, and a fresh seed block.

Suite after gate restoration: failure set **identical** to the recorded
baseline (`09_suite_after_gate_restored.txt`).
