# Low-Ply T1j Qualification — CARD ONLY, NOT AUTHORIZED, NOT RUN

**Status: FROZEN CARD. Nothing has run.** No JVM started, no model loaded, no
seed drawn or registered, no D1 retry, no confirmation data opened, nothing
trained, nothing pushed. This document and its frozen prefix list are the whole
deliverable; the runner does not exist.

**Gate (to be written when execution is authorized):**
`LOWPLY_QUALIFICATION_AUTHORIZED = False`, its **own** constant. It must not read
`D1_EXECUTION_AUTHORIZED`, `L0_EXECUTION_AUTHORIZED` or `SCREEN_AUTHORIZED`: one
gate must never be openable by opening another.

**Scope its eventual authorization may permit: Java and the pinned toolchain
only.** Still barred, and enforced by construction rather than by intention —
**no incumbent model load, no seed action of any kind, no D1 retry, no
confirmation data, no training, no push.** It needs **no seed interval at all**,
because it never invokes the incumbent and therefore never derives an RNG stream.

---

## 1. The question

**Does T1j complete both frozen depths, return a legal move, and bind its
replayed and searched positions, at plies 1–5?**

The single authorized D1 run VOIDed with `exit 3` — E4Preflight's own
`failures != 0` — and the reason was lost because the transcript was discarded
(since repaired at all three call sites). One live possibility is that a
requested depth did not complete at a very low ply. **It is unestablished**, and
this qualification exists to establish or refute it before anyone proposes
amending §12.1.

⚠ **This is not a D1 retry and produces no D1 result.** It queries T1j alone.
Nothing here compares engines, and nothing here may be read as evidence about
the incumbent.

## 2. What is frozen — the exact retained prefixes

`evidence/2026-08-28-t1j-lowply-qualification/01_frozen_prefixes.json`
sha256 `a9054cb2d56cf75f554292e47ac7e5b86a0c74f54b255ee9147fb65f92207acf`

Derived read-only from the D1 run's own input
(`02_positions.json`, sha256 `d5a3cdfa58844451ba21e0fb23781c6aedbda9ad3c239f1c83ea99c3e3d037e3`),
by taking **every position the frozen §12.1–12.3 rule already retained at
ply ≤ 5** and deduplicating by canonical digest, earliest by `(task_id, ply)` —
the same total order §12.2 uses. **Nothing is re-selected**; this is a filter of
an existing frozen set, so it cannot smuggle in a new selection rule.

**9 prefixes. 9 rows at ply ≤ 5, 9 distinct digests, so the dedup removed 0** —
applied anyway, and recorded as a no-op rather than skipped.

| ply | task | prefix | digest |
|---:|---|---|---|
| 1 | `l0match-000-…-o1_center-t1j_red-r0` | `(11,11)` | `487df111c9db` |
| 3 | `l0match-000-…-o1_center-t1j_red-r0` | `(11,11) (12,13) (13,12)` | `470721202fb3` |
| 5 | `l0match-000-…-o1_center-t1j_red-r0` | `… (10,13) (12,10)` | `ea75792355b7` |
| 1 | `l0match-016-…-o3_low-t1j_red-r0` | `(15,11)` | `69c2875679f3` |
| 3 | `l0match-016-…-o3_low-t1j_red-r0` | `(15,11) (12,12) (13,10)` | `4fba47bfca43` |
| 5 | `l0match-016-…-o3_low-t1j_red-r0` | `… (14,14) (11,11)` | `e41208a808b0` |
| 1 | `l0match-024-…-o4_high-t1j_red-r0` | `(8,12)` | `7c326873cac4` |
| 3 | `l0match-024-…-o4_high-t1j_red-r0` | `(8,12) (11,11) (10,13)` | `5e2c1f8ab19e` |
| 5 | `l0match-024-…-o4_high-t1j_red-r0` | `… (9,9) (12,12)` | `14b23d6fe016` |

**Plies 1, 3 and 5 only — no even plies, and that is not an omission.** §12.1
selects plies where **our incumbent is to move**; in the `t1j_red` arm T1j is
red and moves at even plies, so ours moves at odd ones. All nine are `t1j_red`,
across three openings (`o1_center`, `o3_low`, `o4_high`) and three tasks.

⚠ **The cohort therefore says nothing about plies 2 and 4.** A result here
generalises to the low plies D1 would actually query, not to "low ply" in
general.

## 3. Process budget — 36 queries, 45 helper launches, 46 Java processes

Three different counts, and they are not interchangeable:

| count | value | what it is |
|---|---:|---|
| **T1j queries** | **36** | 9 prefixes × 2 depths (`mdPly` 3 and 6) × 2 invocations. **This is the budget**, and it is what a query ceiling bounds. |
| **T1j replay-helper launches** | **9** | one E3bDump replay per prefix, for the E3b binding. Not queries. |
| **T1j helper launches** | **45** | 36 + 9 — every process that runs the T1j jar. |
| **Java/JDK process launches** | **46** | 45 + **one `javac`** compiling the helper once, up front. |

🔑 **`javac` is a JVM process too.** An earlier draft of this card said
"45 JVM invocations", which either under-counts by one or silently means
"launches of the T1j helper" — and the reader cannot tell which. A count whose
unit is ambiguous is not a frozen number. The compilation is a single process
(`compile_helper` invokes `javac` once over all four sources), it happens before
the deadline's first stage check, and it is bounded by the whole-run supervisor
rather than by any per-query timeout, because the adapter's `compile_helper`
takes no timeout parameter at all.

**The 36-query budget is unchanged by this correction**: replays and compilation
were never queries, and a query ceiling was never counting them.

Two invocations per depth, as **separate `query(..., repeats=1)` calls**, for
§12.7's reason and no other: each is a fresh JVM and therefore an independently
drawn `Zobrist` salt. 🔴 **`repeats>1` is PROHIBITED** — it rebuilds inside one
JVM, reuses that process's single salt, and so agrees by construction.

## 4. Cost basis, and a discrepancy I will not inherit

Measured wall-clock per query from the E4 preflight's own record,
`2026-08-25-t1j-e4-preflight-attempt4/03_results.jsonl` (6 query rows per depth;
`wall_ms` includes JVM startup):

| depth | min | median | **max** |
|---:|---:|---:|---:|
| 3 | 121.7 | 144.2 | **215.7** |
| 6 | 202.6 | 587.0 | **2734.5** |

🔴 **§12.9's cost table does not match this record, and I am recording that
rather than reusing its numbers.** It lists depth 3 as `121` — the **minimum**
observed (121.65) — and depth 6 as `2749`, which **appears nowhere in the
results file**; the maximum there is 2734.5. Mixing a minimum with a
near-maximum under one heading of "measured ms/query" understates one depth and
overstates the other. This does not invalidate §12.9, which called its own
subtotal indicative and not a promise, and D1's 90-minute cap has ample margin
either way — but this card uses **maxima throughout, and says so**.

**Indicative T1j subtotal:** 9 × (2 × 215.7 + 2 × 2734.5) = **53,105 ms ≈ 53 s**.

**It excludes** the 9 replays (E3bDump replay cost is **not measured anywhere in
the record**), the single `javac` compilation, output and fsync.

⚠ **And one effect that could push the other way.** E4 measured positions of
6–14 plies; these are 1–5. A sparser board has a **larger branching factor**, so
a fixed-depth search may be **slower** here than E4 ever observed. The subtotal
is a basis for a cap, not a prediction.

## 5. Frozen limits — 120 s per query, 15 minutes whole run

| limit | value | source |
|---|---|---|
| per-call timeout | **120 s** | the qualified E4 preflight limit, `docs/superpowers/2026-08-25-t1j-e4-preflight.md:40` — passed at **every** call site, replay included |
| whole-run wall clock | **900 s (15 min)** | ≈17× the 53 s subtotal, generous margin for the branching caveat, and still a hard bound |

**Neither is redundant, and the arithmetic shows it.** 36 × 120 s = 4,320 s =
**72 minutes**, nearly five times the cap: the per-call timeout alone does not
bound the run. Conversely the cap alone would let one hung call consume the
whole budget with no indication of which call hung.

**Mechanics, as repaired:** the deadline is monotonic, started **before** helper
compilation; the SIGALRM supervisor arms **from that same clock's remaining
time**, so the enforced and reported windows share one origin; and it refuses an
unstarted deadline or a non-positive remaining, because `setitimer(…, 0)`
*disables* the timer.

## 6. Output location

`docs/superpowers/evidence/2026-08-28-t1j-lowply-qualification/`

| file | content |
|---|---|
| `01_frozen_prefixes.json` | **exists** — the frozen input above |
| `02_prerun_verification.txt` | gates, toolchain hashes, limits, input sha256 |
| `03_run.sh.txt` | the exact invocation |
| `04_stdout.txt` / `05_stderr.txt` / `06_exit.txt` | the run's own streams |
| `07_lowply_records.json` | the record — **create-only, fsynced, written once** |
| `08_full_suite_after_gate_restored.txt` | the suite, gate back to `False` |
| `A_MANIFEST.sha256.txt` | hashes of all of the above |

## 7. Outcome vocabulary

- **`PASS`** — all 36 queries complete their requested depth, return a move legal
  in *both* engines, present a clean postcondition surface with the exact
  reflection count, and bind their replayed and searched positions; and the two
  invocations at each depth agree.
- **`FAIL`** — any of those does not hold. **This is a RESULT, not an abort**: it
  is the finding that would inform whether §12.1 should exclude low plies. The
  run completes and records every reply.
- **`VOID`** — instrument failure: identity mismatch, timeout, deadline breach,
  or a refusal that is about the harness rather than about T1j.

🔑 **The FAIL/VOID distinction is the point of this card.** D1 conflated them:
a T1j reply that did not complete its depth aborted the whole run, so the
observation could not be recorded. Here, T1j failing to complete is **the
measurement**.

## 8. What this card does NOT decide

It does not amend §12.1. It does not reserve a seed interval. It does not
authorize itself. Only its **results** can inform whether selection should
change and whether a fresh D1 interval is justified — and both remain separate
decisions.
