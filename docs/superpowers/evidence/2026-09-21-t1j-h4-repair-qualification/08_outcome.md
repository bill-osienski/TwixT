# H4 REPAIR QUALIFICATION — RAN ONCE, verdict `CLEAN`

```
exit 0 · 2026-09-22T16:33:57 → 16:34:08 -0400 · 11 s of a 1800 s deadline
72/72 subprocesses (56 queries + 16 replays) · 16 positions · 72 observations
verdict CLEAN · matrix 3cc14ca9…d22f27cd (the card's pin) · record 05_qualification.json
```

**Run exactly once, as authorized.** Gate opened in its own one-line commit
(`fa6fff0`), the run launched once as a fresh subprocess, and the gate was
restored by the wrapper's EXIT trap to the reviewed file (`796f07a`),
byte-identical — read back from source, not asserted: derived gate count **13**,
open gates **[]** (`07_gate_after.txt`). No Java process survived.

Before the gate opened, `00_prerun_verification.txt` recomputed every input:
source pin, derived-matrix pin (16 rows, 1002 bytes), caps 56 + 16 = 72, jar pin,
JDK identity, create-only destinations absent. It also ran two harmless checks,
stated here so nothing is hidden: a **javac-only** compile into a throwaway
directory (exit 0; no T1j code executes), and the **exact run command with the
gate still closed** — exit 5, nothing created — which qualified the command line
as a fresh subprocess before the one real launch.

⚠ Unlike §4A's precedent, `00` was **not** committed with the gate: the
authorization required the gate-open commit to be the one line alone.

---

## 1. Which routine answered — by ply

| ply | queries | classification | exit / failures |
|---:|---:|---|---|
| 0 | 5 | `native_initial_first` ×5 | 3 / 1 |
| 1 | 15 | `native_initial_second_to_fourth` ×15 | 3 / 1 |
| 2 | 15 | `native_initial_second_to_fourth` ×15 | 3 / 1 |
| 3 | 15 | `native_initial_second_to_fourth` ×15 | 3 / 1 |
| 4 | 3 | `searched` ×3 | 0 / 0 |
| 5 | 3 | `searched` ×3 | 0 / 0 |

* **No search at plies 0–2** — 35 of 35 native, as the card's amendment derives.
  The plies 0–2 STOP had nothing to fire on.
* **Ply 3 answered natively all 15 times.** The `-1,-1` sentinel, the only route
  to a search at ply 3, was not reached in these queries.
* ⚠ **`native_initial_fifth_or_more` WAS NOT OBSERVED.** At all six ply-4/5
  positions `fifthOrMoreMove()` returned null and the search ran. The three ply-5
  positions are the same ones §4A queried; the three ply-4 positions are new. The
  routine has still **never been observed answering** anywhere in this programme,
  and its classification remains exercised only by the constructed-telemetry
  baseline (card §7.1). This is **not** a stop: plies 4–5 may legitimately
  produce either classification (card §4), and both outcomes are recorded.

## 2. The contracts — all held

| contract (card §5 / §4.1) | observed |
|---|---|
| legal in T1j's own report **and** in our engine | 56/56 |
| one same-process dump, coherent with our position | 56/56 queries; 16/16 replay final states |
| exactly one `PROC` per jvm | 72 lines from 72 **distinct** pids |
| reflection: opt-in query / replay | `refl_n` 4 on 56/56 · 1 on 16/16 |
| `MatchData` read back as frozen | 56/56: `pieRule=false xsize=24 ysize=24 ystarts=true identity=true` |
| native reply = exit 3 + failures 1 | 50/50 |
| completed search = exit 0 + failures 0, depth 6, `currentMaxPly=7` | 6/6 |
| replay = exit 0 + failures 0, `ply+1` blocks | 16/16 |
| clean safety surface: no throw, 0 windows, 0 frames, headless, prefs intact | 72/72 |

### Independently re-checked from the raw stdout in the record

The verdict is the runner's; these figures were **re-derived** after the run by
re-parsing every stored stdout with the adapter's parsers and recomputing each
position in our engine — not read off the record's summary fields:

```
independent re-check failures: NONE
PROC lines: exactly 1 in each of 72 stdouts;  distinct pids: 72 of 72
DUMP COHERENCE: 56 query dumps (exactly 1 each) + 16 replay final states
                vs our engine -> divergences: NONE
helper sources in the record == committed files: True
```

## 3. Realized moves — REPORTED, NOT JUDGED

Card §3: five is not a sample size; there is **no diversity threshold, no support
test and no probability estimate**. T1j coordinates `(x, y)`:

| cell | realized (5 fresh JVMs) |
|---|---|
| ply 0, empty board | (13,12) (9,10) (10,9) (10,10) (9,10) |
| ply 1, o1_center | (11,18) (11,18) (11,16) (11,16) (11,18) |
| ply 2, o1_center | (6,12) (5,12) (6,12) (5,12) (7,12) |
| ply 3, o1_center | (12,6) (12,6) (12,6) (12,6) (12,7) |
| ply 1, o3_low | (11,6) (11,7) (11,7) (11,7) (11,6) |
| ply 2, o3_low | (18,12) (17,12) (19,12) (17,12) (19,12) |
| ply 3, o3_low | (10,6) (10,5) (10,5) (10,7) (10,6) |
| ply 1, o4_high | (12,17) (12,17) (12,15) (12,15) (12,15) |
| ply 2, o4_high | (17,11) (18,11) (17,11) (18,11) (17,11) |
| ply 3, o4_high | (13,18) (13,18) (13,17) (13,17) (13,16) |

Searched, one query each: o1_center ply 4 (9,16), ply 5 (11,9); o3_low ply 4
(11,10), ply 5 (11,7); o4_high ply 4 (12,12), ply 5 (12,16).

All five ply-0 moves lie in `x, y ∈ 9..14`, the central region the no-pie branch
of `firstMove()` is derived to draw from. That is **consistent with** the
injected `mdPieRule=false` being the operative setting, alongside the readback;
it is an observation, not a pass condition.

## 4. Scope

**Establishes**, for the 16 frozen positions at depth 6, under the frozen
fresh-JVM lifecycle, through the opt-in `h4query` path, against the pinned jar
`53ec95e4…` with helpers compiled from the committed sources: every reply was a
legal, board-coherent move; which routine answered is identifiable; the PROC,
reflection and exit contracts held; and the injected `MatchData` read back as
frozen.

**Does NOT establish**: anything outside these 16 positions or at another depth;
that `fifthOrMoreMove()` can answer; that ply 3 can reach its sentinel; the
default query path's behaviour (not exercised here — its `refl_n = 3` is pinned
by tests only); any opening probability; **anything about T1j's strength**. No
game was played, no seed drawn, no score computed. Per the card, these results
**may not be pooled** with §4A, H1, H2, H3 or L0.

## 5. What this decides, and what it does not

The card's §7 precondition for §4B — **a CLEAN result, not a partial or a
repaired one** — is met. That is a precondition, **not an authorization**: §4B
was not part of this step and remains unauthorized pending a separate decision.
Seeds, games, strength aggregation and push were not touched.

Suite after gate restoration: failure set **identical** to the baseline measured
on the clean tree at `3902bf5` (`09_suite_after_gate_restored.txt`).
