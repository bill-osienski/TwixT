# H4 §4A RAW CAPABILITY CHARACTERIZATION — RAN ONCE, verdict `STOP_ZERO_LENGTH_REFUSED`

```
exit 0 · 2026-09-21T16:15 → 16:16 · 3.4 s of a 900 s deadline
20/20 subprocesses spent · 10 positions · 20 observations · record written
verdict STOP_ZERO_LENGTH_REFUSED · branches fired: ZERO_LENGTH_REFUSED, INSTRUMENT_FAILURE
```

**A STOP IS A RESULT, and it ends §4A.** Per the authorization, nothing was
repaired, retried or improvised after it. The gate was opened in its own commit
(`ae60669`), the run executed once, and the gate was restored to `False`
immediately afterwards — read back from source, not asserted: derived gate count
**12**, open gates **[]**.

---

## 1. The two questions §4A existed to answer

### ✅ Q1 — Is a searched-position dump emitted when T1j never searches? **YES.**

All six non-searching queries (board-plies 1 and 3, `usealphabeta=False`,
`currentMaxPly=0`) emitted **exactly one** searched-position dump, **coherent
with our position** — zero divergences — **from the same process**, alongside a
**legal move**:

| ply | prefix | move returned | dumps | divergences | same-process | legal in our engine |
|---:|---|---|---:|---|---|---|
| 1 | `o1_center` | `[17, 11]` | 1 | none | yes | yes |
| 3 | `o1_center` | `[6, 12]` | 1 | none | yes | yes |
| 1 | `o3_low` | `[6, 11]` | 1 | none | yes | yes |
| 3 | `o3_low` | `[7, 10]` | 1 | none | yes | yes |
| 1 | `o4_high` | `[17, 12]` | 1 | none | yes | yes |
| 3 | `o4_high` | `[16, 13]` | 1 | none | yes | yes |

🔑 **§4B's board-coherence requirement is REACHABLE for non-searching replies.**
The card flagged the opposite possibility — that no dump would exist and the
requirement would be unmeetable — and named the honest repair. **That repair is
not needed.** `len(dumps) == 1` holds on the fallback path, so §4B's classifier
can require it without weakening anything.

### 🔴 Q2 — Does the helper accept a zero-length position? **THE TWO PATHS DIVERGE.**

| path | Java main | ply-0 result |
|---|---|---|
| **replay** | `E3bDump` | ✅ **ACCEPTED** — exit 0, one ply block, clean surface |
| **query** | `E4Preflight` | 🔴 **THREW** — exit 3, **no query record at all** |

This is exactly why the card insisted both paths be characterized and that
"neither answers for the other."

**The throw, verbatim from the record:**

```
THREW: java.lang.NullPointerException: Cannot read field "mdPieRule" because
the return value of "net.schwagereit.t1j.Match.getMatchData()" is null
    at net.schwagereit.t1j.InitialMoves.firstMove(Unknown Source)
    at net.schwagereit.t1j.InitialMoves.initialMove(Unknown Source)
    at net.schwagereit.t1j.FindMove.computeMove(Unknown Source)
    at net.schwagereit.t1j.E4Preflight.queries(E4Preflight.java:155)
```

⚠ **STATE THIS CAREFULLY.** The empty board routes T1j into a *dedicated*
first-move path, `InitialMoves.firstMove`, which reads match settings
(`mdPieRule`) from `Match.getMatchData()`. Under `E4Preflight` that object is
**null**.

**So this is a failure at the HELPER boundary, not a demonstration that the
engine cannot open a game.** T1j plainly has a first-move routine; our
preflight harness never initializes the match data it requires. §4A is not
authorized to decide what follows from that, and does not.

Note also that the ply-0 query **did** emit a coherent empty-board dump
(0 pegs, 0 bridges, 528 legal cells — matching our engine exactly) *before*
throwing. The position was built; computing a move is what failed.

---

## 2. An unanticipated finding: `E3bDump` emits no `PROC` line

| path | observations | PROC lines each |
|---|---:|---:|
| query (`E4Preflight`) | 10 | **1** |
| replay (`E3bDump`) | 10 | **0** |

🔴 **This bears directly on the card's §3.2.** That section asserts
*"replay-process count == final ply count + 1"*, to be checked from `PROC`.
**On the replay path that assertion is not satisfiable with the helper as it
stands** — there is nothing to count. The query path carries PROC exactly as
expected, 1 per jvm, 10 for 10.

§4A reports this and stops. It is a fact about the instrument, recorded for
whoever amends the card.

---

## 3. Every branch that fired, and why the headline is the refusal

**`STOP_ZERO_LENGTH_REFUSED`** — 1 reason:
* `empty_board@ply0 query`: the zero-length position yielded no query record.

**`STOP_INSTRUMENT_FAILURE`** — 13 reasons:
* `empty_board@ply0 query`: safety surface not clean — **the helper threw**;
  1 reflective access where exactly 3 were expected; `failures=1` on a query
  that reports `completed`, where the incomplete-search exemption does not apply.
* ten × `… replay: 0 PROC lines, expected exactly 1 per jvm`.

The precedence puts the refusal first **by design**: a ply-0 refusal routinely
drags unclean output behind it, and calling that "the instrument failed" would
relabel the finding §4A exists to make. Both branches are recorded, so nothing
is hidden by the choice of headline.

---

## 4. What ran cleanly — the negative space of the result

Plies 1, 3 and 5 behaved exactly as the 2026-08-31 low-ply record predicts,
which is the check that this run's instrument was sound:

* **ply 5**: `searched`, exit 0, `failures=0`, clean surface, one coherent dump.
* **plies 1 and 3**: `incomplete`, `usealphabeta=False`, `currentMaxPly=0`,
  exit 3, `failures=1`, **clean safety surface** — the exemption behaving as
  designed, and the reason `PROCEED` was reachable at all.
* **every replay at plies 1/3/5**: exit 0, correct arity (`ply+1` blocks: 2, 4,
  6), no final-state divergence, clean surface.

So the refusal at ply 0 is not an instrument that was broken throughout. It was
working everywhere else in the same run.

---

## 5. Scope — what this record does and does not establish

**Establishes**, for board-plies 0/1/3/5 at depth 6 on the frozen prefix set,
under the frozen fresh-JVM lifecycle:
* a non-searching query emits one coherent same-process dump;
* `E3bDump` accepts a zero-length position; `E4Preflight` throws on one;
* `E3bDump` emits no `PROC`; `E4Preflight` emits one per jvm.

**Does NOT establish**: anything about plies 2 or 4; any other depth; anything
about T1j's strength; anything about whether the engine could open a game given
properly initialized match data. **No game was played, no seed drawn, no score
computed.** Ply 0 is the only red-to-move position here; plies 1/3/5 are all
black-to-move.

---

## 6. What this decides, and what it does not

Per the card's frozen §4A.1 branches, `STOP_ZERO_LENGTH_REFUSED` means:

> **STOP. Arm B is blocked. Any CLI grammar repair requires separate review.**

§4A is complete and closed. **It does not authorize §4B**, and the card's
recorded repair options — an explicit `move_count=0` representation, and never a
dummy opening move — remain **separately reviewable proposals, not decisions
taken here.** The `getMatchData()` finding is new information that a reviewer
may weigh; §4A neither acts on it nor recommends a course.
