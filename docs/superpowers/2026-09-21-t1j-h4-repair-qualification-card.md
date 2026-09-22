# H4 REPAIR QUALIFICATION — CARD

**Written 2026-09-21. DESIGN ONLY.** Nothing is implemented, no gate exists,
nothing runs, no research seed is reserved.

🔴 **SEPARATELY NAMED. THIS IS NOT A §4A RETRY OR CONTINUATION.** §4A is closed
at `STOP_ZERO_LENGTH_REFUSED`, and its verdict stands on the helper as it was.
The helper's behaviour changes here, so this qualification carries its **own
name, own gate, own create-only destination, and its own frozen stop
conditions**. Its results **may not be pooled with §4A's**.

> **What it must establish.** That the repaired helper returns a **legal,
> board-coherent move from every position H4 will actually query**, that
> **which routine answered is identifiable**, and that the **process and
> reflection contracts hold** — with no diversity claim, no probability
> estimate, and no strength evidence of any kind.

Depends on: `2026-09-21-t1j-h4-matchdata-repair-design.md` (both changes), which
must be implemented and reviewed **before** this card is frozen.

---

## 1. 🔴 THE MATRIX IS PINNED BY ITS DERIVED FORM, NOT BY ITS SOURCE

**This is the §4A fail-open, one level up, and it must not be repeated.**

§4A pinned `01_frozen_prefixes.json` by sha256 and then built its matrix from
it. That was sound only because the matrix was a **filter** over pinned rows.
Here the matrix is a **derivation**: plies 2 and 4 are produced by **truncating**
each family's ply-5 sequence.

🔴 **The source hash authenticates the source sequences. It says NOTHING about
the 16 rows derived from them.** Truncation code could drop a row, duplicate
one, emit them out of order, or truncate to the wrong length, and
`FROZEN_PREFIXES_SHA256` would keep passing while the qualification ran a
different matrix.

**Therefore the complete derived matrix is pinned in its own right:**

| pinned | why |
|---|---|
| **content** — every row's `(family, ply, prefix)` | a swapped position is a different question |
| **order** | observations carry a monotonic ordinal (§4A's own lesson) |
| **row count = 16** | a dropped or duplicated row changes the coverage |
| **sha256 over the canonical serialization** | one value that all four of the above must reproduce |

The runner **recomputes the derivation and compares the whole canonical matrix
against that pin before compilation**, exactly as §4A's corrected
`build_matrix` compares against its own. Both pins are checked: the **source**
hash proves the sequences are the frozen ones, and the **derived** hash proves
the truncation produced the matrix this card froze.

🔑 **Nesting is what makes plies 2 and 4 free of new input**, and it is verified
rather than assumed: each family's ply-1 prefix must be a prefix of its ply-3,
which must be a prefix of its ply-5. If that fails, the derivation is invalid
and the qualification does not start.

✅ **CHECKED 2026-09-21, against the pinned file.** All three families nest
exactly (`ply1 ⊂ ply3 ⊂ ply5`, each ply-5 sequence 5 moves long), and all six
derived ply-2 / ply-4 positions **replay legally in our engine**. The
derivation is sound on today's pinned input; the runner must still re-check it,
because a check performed once in a card is not a check performed at run time.

---

## 2. The base matrix — 16 positions

The **empty board**, plus **plies 1–5** from each of the three frozen families
(`o1_center`, `o3_low`, `o4_high`).

| | positions |
|---|---:|
| empty board | 1 |
| 3 families × plies 1, 2, 3, 4, 5 | 15 |
| **total** | **16** |

🔴 **WHY NOT §4A's 0/1/3/5.** §4A states in terms that it speaks for **no ply 2
or 4**. In H4, **T1j Red moves at plies 0, 2, 4** and **T1j Black at 1, 3, 5**,
so repeating §4A's matrix would leave **half of Arm B's early moves
unqualified** — and Arm B is the arm the whole repair exists to enable.

✅ **VERIFIED, not argued.** Replaying the derived rows in our engine: every
ply-0/2/4 position is **red to move** and every ply-1/3/5 position is **black to
move**. So the coverage is:

| T1j's role | positions |
|---|---:|
| **RED** (plies 0, 2, 4) | **7** — the empty board + 2 per family |
| **BLACK** (plies 1, 3, 5) | **9** — 3 per family |

§4A's matrix gave the red role **exactly one** position (ply 0). This gives it
seven, which is the whole point of the correction.

---

## 3. Repetitions — frozen, and for OPERABILITY only

| cells | repetitions | rationale |
|---|---:|---|
| **randomized**: empty board + each family at plies **1, 2, 3** (**10 cells**) | **5 fresh-JVM queries** each | plies 0–3 dispatch to `firstMove()` / `secondToFourthMove()`, both of which construct a fresh unseeded `Random`. Five follows the prior fresh-process precedent |
| **deterministic**: each family at plies **4, 5** (**6 cells**) | **1 query** each | `fifthOrMoreMove()` contains no `Random` |
| **every base position** | **1 replay** each | the replay path is unaffected by opening randomness |

### Derived caps

```
queries  = 10 randomized cells × 5  +  6 deterministic cells × 1  =  56
replays  = 16 base positions × 1                                  =  16
                                                     TOTAL        =  72
```

🔑 **DERIVED, never hand-typed.** The runner computes these from the matrix it
actually built, and a test asserts the arithmetic — §4A's budget bug was exactly
a hand-typed constant standing in for a derivation.

🔴 **FIVE IS NOT A SAMPLE SIZE.** The repetitions exist to show the routine
**keeps returning legal, coherent moves across processes**. There is **no
diversity threshold, no support test, and no probability estimate.** Realized
moves are **reported**; the number of distinct ones is **never** a pass
condition, in either direction.

---

## 4. The exact source classifier — frozen before any output

Recorded per query, from telemetry the repaired helper already emits:

| ply | telemetry | classification |
|---:|---|---|
| **0** | `usealphabeta=false`, `currentMaxPly=0` | `native_initial_first` |
| **1–3** | `usealphabeta=false`, `currentMaxPly=0` | `native_initial_second_to_fourth` |
| **4–5** | `usealphabeta=false`, `currentMaxPly=0` | `native_initial_fifth_or_more` |
| any | `usealphabeta=true`, `currentMaxPly=7`, completed depth **6** | `searched` |
| — | **every other combination** | 🔴 **STOP** |

**`currentMaxPly=7` is `depth+1`**, as the low-ply record shows for a completed
depth-6 search. The classification is keyed on the **ply dispatch plus the
telemetry**, so it names **which routine answered** rather than inferring a
narrative from an exit code.

⚠ **Plies 4 and 5 may legitimately produce EITHER classification.**
`fifthOrMoreMove()` may return a native move **or null**, and null falls through
to search. §4A observed null at three ply-5 positions; **that is three
positions, not a property of ply 5.** Both outcomes are recorded and neither
stops the run.

---

## 5. The contracts that must hold

* **every returned move legal in T1j's own report AND legal in our engine**, and
  **coherent with its same-process dump**;
* **exactly one `PROC` per query JVM and per replay JVM** — 56 + 16 = 72,
  derived, which is what repair Change 2 exists to make checkable;
* **reflection**: query **`refl_n = 4`** on the opt-in path, replay
  **`refl_n = 1`**, and 🔴 **the default query path stays pinned at 3** — a
  control must prove a default caller is unchanged;
* **the injected `MatchData` read back and recorded** — `mdPieRule=false`
  (operative), `mdXsize=24`, `mdYsize=24`, `mdYstarts=true` (recorded,
  not operative);
* **clean safety surface** on every call: no throw, `windows=0`, `frames=0`,
  headless, preferences unchanged.

---

## 6. It STOPS on

Frozen before any output is seen, as §4A's branches were:

* any **null or illegal** move, in either engine;
* any **board divergence** from the same-process dump;
* any **dirty safety surface**;
* any **lifecycle mismatch** — a PROC count that is not 1 per JVM, or a derived
  total that is not 72;
* any **telemetry combination outside §4's table**, including a **native-initial
  result claimed at ply ≥ 6**, where `initialMove()` returns null by dispatch so
  no such reply can exist;
* the **derived-matrix pin** failing, or the **nesting check** failing (§1).

A stop **ends the qualification**. No repair, no retry, no improvisation —
§4A's precedent, and the reason its result is trustworthy.

---

## 7. 🔴 Only a CLEAN result authorizes §4B

Not a partial one, and not one repaired after the fact.

**This card produces no strength evidence.** No research seed is drawn, no game
is played, no score is computed, and nothing here may be pooled with §4A, H1,
H2, H3 or L0.

---

## 8. What this card does not do

It writes no code, opens no gate, reserves no seed and runs nothing. It does not
authorize the repair implementation — that is the repair design's own review —
and it is **not frozen** until the two changes are implemented and reviewed,
because a qualification must be written against the helper it will actually run.

### Still owed before freezing

* the **derived matrix's own sha256**, computed from the canonical
  serialization once the truncation is written;
* the qualification's **gate name** and **create-only destination**;
* confirmation that **`refl_n = 4`** is what the repaired helper actually
  reports — **re-derived, not assumed** (repair design §2.4).
