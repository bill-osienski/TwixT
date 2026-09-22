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

🔴 **THIS CARD IS THE AUTHORITY, AND IT IS FROZEN FIRST.** An earlier draft had
the sequencing backwards — it said the card could not be frozen until the repair
was implemented and reviewed. **That is the wrong way round.** A specification
completed *from* its implementation is not a specification; it ratifies whatever
was built.

So: the card freezes, and then **implementation either CONFORMS to it or returns
here for an explicit design amendment.** It does not get to settle an open
question by being written.

Depends on `2026-09-21-t1j-h4-matchdata-repair-design.md` (both changes) for
**what** is being repaired — not for permission to specify how it is qualified.

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

### ✅ THE PIN, FROZEN NOW — the runner REPRODUCES it, never defines it

The canonical form is specified here, so no code is needed to fix it:

* **UTF-8 compact JSON array**, separators `,` and `:`
* keys in the order **`family`, `ply`, `prefix`** — a declared order, never
  `sort_keys`
* **`empty_board` first** (`ply` 0, `prefix` `[]`)
* families in the order **`o1_center`, `o3_low`, `o4_high`**
* within each family, **plies ascending 1–5**, each `prefix` the first *ply*
  moves of that family's ply-5 sequence
* **no trailing newline**

```text
H4_REPAIR_MATRIX_SHA256 =
  3cc14ca99935af511614742feafe6b3966ce8da83dc2789a103f98d3d22f27cd
```

**16 rows, 1002 bytes.** ✅ **Independently reproduced 2026-09-21** from the
pinned source file via the form above, matching the value derived separately in
review — two derivations, one value, agreeing before it was recorded.

Boundary excerpts, so a reproducer can localize a mismatch instead of only
seeing a wrong digest:

```text
first 90  [{"family":"empty_board","ply":0,"prefix":[]},{"family":"o1_center","ply":1,"prefix":[[11,
last  50  ,"prefix":[[8,12],[11,11],[10,13],[9,9],[12,12]]}]
```

🔑 **The runner must REPRODUCE this serialization, not define it.** That is the
whole point of fixing it in the card: an implementation that computes its own
canonical form and then hashes it has pinned nothing — it has recorded what it
happened to build.

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

### 🔴 AMENDMENT 2026-09-22 — `searched` IS IMPOSSIBLE AT PLIES 0, 1 AND 2

⚠ **CORRECTED THE SAME DAY.** The first version of this amendment (`3902bf5`)
made **ply 0** the only STOP and left ply 1 **permissive as underived**. That
rested on a misread, set out under *"What the first version got wrong"* below —
kept there, not rewritten away. This is the corrected rule.

Under the frozen **24×24 no-pie** configuration, `InitialMoves` answers **every**
query at plies 0, 1 and 2 natively, so no search can run there. **A completed
search at any of those plies contradicts the pinned jar's dispatch. STOP.** This
narrows the "any" in the §4 table's `searched` row, and in §7.1's `searched`
baseline, to **plies 3 and above**.

**Derived, not assumed** — `javap -c` on the pinned `t1j.jar`:

* `FindMove.computeMove()` calls `initialMove()` **first** (offset 12) and, when
  it gets a move, returns it without searching (`ifnull 40`, `areturn` 39).
  `initialMove()` keeps a move only if `getX() >= 0`.
* **ply 0** — `firstMove()` contains **no `aconst_null`**, and its
  `mdPieRule=false` branch always constructs `new Move(x, y)` with
  `x = Xsize/2 + nextInt(Xsize/4) - (Xsize/4)/2` — on a 24-wide board
  `12 + [0..5] - 3` = **9..14**, always ≥ 0.
* **plies 1 and 2** — `secondToFourthMove()`'s only `-1,-1` stores are at offsets
  **495–498**, and the only way in is falling through `492: if_icmpgt 499`, on
  the **moveNr == 3** path. At moveNr 1, `292: if_icmpne 499` jumps **over**
  them; at moveNr 2, `442: if_icmpne 499` does. The `Move` built at 499 keeps
  coordinates computed from a placed move (`getMoveX(i)` is `moves.get(i-1)`),
  and no path drives x below 0.

| ply | routine | can it search? |
|---:|---|---|
| **0** | `firstMove()` | 🔴 **no** — always native |
| **1–2** | `secondToFourthMove()` | 🔴 **no** — always native |
| **3** | `secondToFourthMove()` | **either** — the only ply that can reach the `-1,-1` stores |
| **4–5** | `fifthOrMoreMove()` | **either** — it returns a move or null |
| **≥ 6** | — | **searched only** — `initialMove()` returns null by dispatch |

✅ **It agrees with every observation on record.** §4A saw native replies at
plies 1 and 3 and `searched` at ply 5, in all three families. The low-ply run saw
JVM disagreement at plies 1 and 3 — a randomized routine answering — and none at
ply 5.

**The STOP names no cause.** It records that the reply is one the frozen
configuration cannot produce, not why.

🔑 **Plies 3–5 remain legitimately EITHER.** At ply 3 `secondToFourthMove()` may
return the `-1,-1` sentinel, and at plies 4–5 `fifthOrMoreMove()` may return
null; `initialMove()` turns either into null and the search runs.

**Required controls, in §7.1's sense:** a completed search at ply 0, at ply 1 and
at ply 2 must each be shown to **STOP** — **three separate controls**, so that
dropping any one ply from the rule fails a control of its own — and a completed
search at plies 3, 4 and 5 must be shown to be **accepted**, so that the rule
cannot spread.

⚠ **Plies 4 and 5 may legitimately produce EITHER classification.**
`fifthOrMoreMove()` may return a native move **or null**, and null falls through
to search. §4A observed null at three ply-5 positions; **that is three
positions, not a property of ply 5.** Both outcomes are recorded and neither
stops the run.

#### ⚠ What the first version got wrong — kept, not rewritten away

It said:

> ⚠ **PLY 1 IS *NOT* INCLUDED, AND THE REASON IS THAT I COULD NOT DERIVE IT.**
> … `secondToFourthMove()` has a **single `areturn`**, and its dispatch at
> offsets 277–292 reads as *"moveNr == 2 or 3 → compute, else →
> `new Move(-1,-1)`"* — which would make **ply 1 always search** …

1. **The misread.** The jump to offset 499 was read as though execution passed
   *through* the `-1,-1` stores at 495–498 on its way there. **A branch target
   is where execution RESUMES, not a region it traverses:** `292: if_icmpne 499`
   skips them, and locals 2/3 keep the coordinates already computed. The
   contradiction with §4A's native ply-1 replies was the signal, and **when a
   derivation contradicts an observation, the derivation is wrong.** Refusing to
   encode a rule on a reading known to be wrong was right; stopping at the
   contradiction instead of hunting the error cost a review round.
2. **Ply 2 was missed as well.** Review argued ply 1. The same reading of
   `442: if_icmpne 499` makes ply 2 equally impossible, yet the first version
   declared *"Plies 2–5 remain legitimately EITHER"* and pinned it with a
   control — so a search at ply 2 was **accepted**, and the `searched` positive
   baseline **blessed** it, as an earlier baseline had blessed ply 0.
3. **Two of its sentences were not derived.** *"a completed search at ply 0 means
   the injection did not take effect"* — the runner verifies the MATCHDATA
   readback (`identity=true`, `pieRule=false`) **before** it classifies, and §4A
   showed that a missing injection makes `firstMove()` **throw**, not search.
   And *"at those plies [2–5] a native routine may return the `-1,-1`
   sentinel"* — only ply 3 can; at plies 4–5 the route to search is
   `fifthOrMoreMove()` returning **null**.

### 4.1 🔴 EXIT SEMANTICS — the most dangerous detail, and it was missing

The repair does **not** change the helper's completion requirement, and no such
change is authorized. **Therefore a legitimate native-initial reply still looks
like a failure at the process boundary:**

```text
exit = 3   failures = 1   completed = false
usealphabeta = false      currentMaxPly = 0
```

`E4Preflight` sets its `failures` counter when the requested depth did not
complete, and `initialMove()` answering means no depth completed. **That is
expected, and it is exactly the shape D1 mistook for an abort.**

**Frozen rules — nothing else passes:**

| observation | rule |
|---|---|
| exact **native-initial** signature | permit **only** `exit = 3` **and** `failures = 1` |
| **completed search** | require **`exit = 0`** and **`failures = 0`** |
| native signature with **any other failure count** | 🔴 **STOP** |
| searched reply with **any** failure | 🔴 **STOP** |
| **every other** exit / telemetry combination | 🔴 **STOP** |

🔑 **Without this, "a clean result" is ambiguous** — it could be read as "exit 0
everywhere", which would reject every legitimate native reply, or as "ignore the
exit code", which would accept a genuinely broken one. Pinning both directions
is what makes §7's authorization meaningful, and it is the §4A `failures`
exemption lesson stated as an acceptance rule rather than a bug fix.

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

## 7.1 🔴 REQUIRED NEGATIVE CONTROLS — each must be proven to REJECT

A gate, a pin and a classifier that have never been shown to refuse anything are
assertions. Every one of these is required before the qualification may run:

| control | what it must reject |
|---|---|
| **closed gate** | both public entries refuse, **and** the CLI refuses as a **fresh subprocess**, with no class directory created |
| **matrix row dropped** | a 15-row matrix |
| **matrix row duplicated** | 17 rows, or 16 with a repeat |
| **matrix reordered** | the same 16 rows in another order — order is pinned |
| **wrong truncation length** | a ply-2 row carrying 3 moves |
| **invalid source telemetry** | any exit/telemetry combination outside §4 and §4.1 |
| **native reply at ply ≥ 6** | a native-initial claim where `initialMove()` returns null by dispatch |
| **missing `PROC`** | 0 PROC lines on a query or a replay |
| **multiple `PROC`** | 2+ PROC lines from one JVM |
| **`MatchData` readback mismatch** | injected values that do not read back, incl. `mdPieRule` coming back `true` |
| **dirty safety state** | a throw, `windows`/`frames` non-zero, not headless, preferences disturbed |
| **default-path reflection drift** | a default caller reporting `refl_n != 3` |

⚠ **The ply ≥ 6 control is necessarily SYNTHETIC**, because the frozen matrix
ends at ply 5. It is a classifier control driven with constructed telemetry, not
a position in the matrix — and the card says so rather than letting a reader
assume the matrix covers it.

🔑 **And CLEAN-BASELINE controls — FOUR OF THEM, one per accepted
classification.** Every row above is a refusal, so without positive baselines a
classifier tightened until nothing passes would satisfy the whole table while
making the qualification unusable — the same defect, from the other side, as the
`failures` exemption that once condemned every native reply.

🔴 **ONE AGGREGATE "VALID RECORD" IS NOT ENOUGH.** Each accepted classification
must be exercised **individually**, or an unobserved branch stays unreachable
while every control passes:

| baseline | must reach |
|---|---|
| ply 0, `exit 3` / `failures 1` / `usealphabeta=false` / `currentMaxPly=0` | `native_initial_first` |
| ply 1–3, same telemetry | `native_initial_second_to_fourth` |
| **ply 4–5, same telemetry** | **`native_initial_fifth_or_more`** |
| any ply, `exit 0` / `failures 0` / `usealphabeta=true` / `currentMaxPly=7` / completed depth 6 | `searched` |

⚠ **`native_initial_fifth_or_more` IS THE ONE WITH NO PRIOR EVIDENCE AT ALL, and
that is exactly why it needs its own baseline.** §4A queried three ply-5
positions and every one came back `searched` — `fifthOrMoreMove()` returned null
at all three. **This programme has therefore never observed that routine
answering.** A single aggregate baseline would almost certainly exercise
`searched` or a low-ply native reply and leave that branch dead, with the whole
control table still green.

Because the matrix cannot guarantee a ply-4/5 native reply, this baseline is
**driven with constructed telemetry** where necessary — the same
classifier-control status as the ply ≥ 6 refusal, and stated for the same
reason.

---

## 8. What this card does not do

It writes no code, opens no gate, reserves no seed and runs nothing. It does not
authorize the repair implementation — that is a separate authorization.

⚠ **An earlier draft ended here saying the card was "not frozen until the two
changes are implemented and reviewed, because a qualification must be written
against the helper it will actually run."** That sentence is **withdrawn**. It
inverted the dependency: a qualification written against the helper that was
built can only ratify it. The card is written against the helper the design
**specifies**, and a helper that does not match it **fails the card**.

### ✅ FROZEN NAMES

```text
gate         H4_REPAIR_QUALIFICATION_AUTHORIZED
destination  docs/superpowers/evidence/2026-09-21-t1j-h4-repair-qualification/
matrix pin   H4_REPAIR_MATRIX_SHA256 = 3cc14ca9…d22f27cd   (§1)
```

Naming the gate and its create-only destination is **design work, not
implementation**. The gate constant is **created closed** when the code is
written; naming it now is what lets this card be the authority rather than a
description of whatever gets built.

### ✅ `refl_n = 4` IS THE FROZEN EXPECTED CONTRACT

An earlier draft listed it as "still owed — confirmation of what the repaired
helper reports". **That was the reversed sequencing again.** The card **cannot**
wait for a pre-qualification helper run to learn its own acceptance threshold —
and it does not need to:

* **implementation review** must **re-derive four from the repaired source**;
* **qualification execution** must **observe four**;
* a helper reporting anything else **fails this card** rather than amending it.

### Nothing is owed before freezing

The matrix pin, the gate name, the destination, the classifier, the exit
semantics and the negative controls are all fixed above. **This card is ready to
freeze.**

**The next separately authorized action is repair implementation plus tests,
with the gate created CLOSED and no helper execution.**
