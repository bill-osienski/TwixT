# H4 REPAIR DESIGN — `MatchData` initialization and `E3bDump` PROC emission

**Written 2026-09-21. DESIGN ONLY.** Nothing is implemented, no gate exists,
nothing runs, no seed is reserved. This card exists to be argued with.

**Preserves H4.** This is not a retreat from the empty-board study; it is the
separately-reviewed repair that §4A's `STOP_ZERO_LENGTH_REFUSED` requires before
§4B can be contemplated.

> **What the repair must achieve.** `E4Preflight` must be able to ask T1j for a
> move at board-ply 0 **with no new `RightPanel` path and zero windows or
> frames**, and every JVM on both helper paths must identify itself — while
> every existing caller keeps the behaviour it was qualified with.

⚠ Not "without loading a GUI class". `GuiBoard` is **already resident** on every
non-empty query, because `setlastMove()` routes through its static
`getHoleName()`. See the correction in §2.3.

---

## 0. What §4A established, and one thing it did not

From `evidence/2026-09-21-t1j-h4-4a-characterization/` (verdict
`STOP_ZERO_LENGTH_REFUSED`), as amended by its `10_correction.md`:

| | |
|---|---|
| a non-searching query **does** emit one coherent same-process dump | ✅ §4B's coherence requirement is reachable |
| `E3bDump` accepts a zero-length position | ✅ |
| `E4Preflight` **throws** at ply 0 — `Match.getMatchData()` is null | 🔴 the thing to repair |
| `E3bDump` emits **no** `PROC` line | 🔴 the second thing to repair |
| the grammar parsed fine; a coherent empty-board dump was emitted before the throw | `move_count=0` is **retired** (`10_correction.md` §2) |

🔴 **NOT NEW, and the record must not imply otherwise.** The `mdPieRule` /
`getMatchData()` mechanism was identified statically in
`2026-08-24-t1j-e2-blocked.md` §2. §4A is its **first runtime confirmation on
this path**. See `10_correction.md` §1.

---

## 1. The prior art this repair must respect

`2026-08-24-t1j-e2-blocked.md` did not merely name the null. It closed off the
alternatives, and those findings constrain the design:

1. **The public initialization methods are unusable.** `prepareNewMatch()` and
   `updateMatchData()` are the only methods that set `matchData`, and **both
   call `RightPanel.getInstance()`** — a GUI class the authorization forbids.
2. **No subclass route exists anywhere on the engine path.** `Match`, `Board`,
   `FindMove` and `InitialMoves` are each `public final class`; `GeneralSettings`
   has a private constructor and an eager `static final` instance.
3. **Same-package access does not help.** `matchData` — with `boardY`, `boardX`,
   `moves`, `moveNr` and the rest — is `private`.
4. Therefore **reflection into private state is the only non-GUI route** ⚠ *(as
   E2 reasoned it then — see the correction immediately below, which narrows
   this)*, and E2 refused to take it unilaterally: *"a headless path that depends on private
   internals is a materially different claim from one that uses the published
   API — so E2 stops rather than picking between them."*

🔑 **This card is the review E2 deferred.** Choosing reflection is a material
change in what our harness claims, and it is being made explicitly, in writing,
rather than slipped in as an implementation detail.

### ⚠ CORRECTED 2026-09-21 — point 4 held under E2's THEN-CURRENT criterion only

An earlier draft of this card carried point 4 forward as though it still closed
every public route. **It does not, and the card may not claim it does.**

E2's "only non-GUI route" was reasoned under an authorization that forbade
**loading** a GUI class at all. **Attempt 4 relaxed that**, and passed with GUI
classes *resident*:

> **25 distinct class names … loaded**: … `GuiMainWindow` and — new this attempt
> — `GuiBoard` are both **resident**, the latter because `setlastMove()` routes
> through `GuiBoard.getHoleName()`. … T1j can be loaded … and made to return one
> legal move … **with GUI classes resident, no window instantiated**, the host
> preference surfaces untouched
> — `2026-08-24-t1j-e2-attempt4.md`

Under **today's** broader criterion — residency permitted, **zero windows and
frames** — the public initialization route is **not logically closed**. What it
is, is **unqualified**: `prepareNewMatch()` and `updateMatchData()` reach
`RightPanel.getInstance()`, and nothing in the record establishes whether that
instantiates a window.

**Reflection remains the narrowest recommended repair**, and that is the claim
this card makes. It is a judgement about scope and auditability, **not** a proof
that no alternative exists.

### 1.1 🔴 A consequence for the opening that the design must carry

E2 recorded, on the way past:

> `InitialMoves.firstMove()` selects from a **seven-entry opening table** with
> an **unseeded `new Random()`**. The opening-book branch is therefore
> **nondeterministic by construction.**

🔴 **CORRECTED 2026-09-21 — THE SEVEN-ENTRY TABLE IS THE WRONG BRANCH FOR H4.**
An earlier draft of this card carried E2's sentence forward unqualified. It
describes the branch H4 **deliberately does not take.**

`firstMove()` reads `mdPieRule` and forks. Derived from the pinned jar by
disassembly (`javap -c net.schwagereit.t1j.InitialMoves`):

| `mdPieRule` | branch |
|---|---|
| **true** | the **seven-entry table**: `nextInt(7)` over a `new int[7][]`, then two `nextBoolean()` mirrorings off `getXsize()` / `getYsize()` |
| 🔴 **false** — **H4's setting** | **two independent unseeded `nextInt` draws off board size** |

The no-pie branch, read straight off the bytecode at offsets 217–294:

```
r = Xsize / 4 ;  x = Xsize/2 + nextInt(r) - r/2
s = Ysize / 4 ;  y = Ysize/2 + nextInt(s) - s/2
```

On H4's **24×24** board that is `12 + [0..5] - 3` on each axis, so
**x ∈ 9..14 and y ∈ 9..14 — the central 6×6 region, 36 coordinates**, from two
independent draws of a **fresh unseeded `Random`**.

So a repaired ply-0 query does **not** return a fixed move. T1j's opening is
drawn from **that 6×6 region** by **randomness whose state we neither seed nor
record; every realized move is observed and persisted.**

⚠ **CORRECTED 2026-09-21.** An earlier draft said "randomness we neither seed
nor observe". That was wrong in a way that matters: **the realized move IS
observed, and persisted.** What goes unrecorded is the **latent RNG state**.
The estimand already covers the execution/randomization protocol rather than
reproducibility from our seed alone, so an unrecorded latent state is a declared
property of the protocol, not a gap in the evidence.

This is native behaviour and **must not be seeded, replaced or collapsed** — it
is part of the agent H4 measures, and it is real entropy at exactly the ply the
superseded H4 card wrongly assumed had none.

🔴 **AND THE QUALIFICATION MUST NOT TRY TO MEASURE IT.** Do not seed it, do not
collapse repeats, do not require diversity. The qualification establishes
**legality and operability**; it does **not** estimate opening probabilities.
Seven table entries is prior art, not a target.

---

## 2. CHANGE 1 — an opt-in `MatchData` initialization path (BEHAVIORAL)

🔴 **This changes what T1j does. It is not observational, and it must never be
described as though it were.**

### 2.1 Opt-in, default off, existing callers untouched

A new explicit mode. Every current caller — E4, L0, H1, H2, H3, D1, the low-ply
qualification, the runtime requalification and §4A itself — keeps the exact
behaviour it was qualified with, and a control must prove it.

### 2.2 What it must freeze, explicitly

Match settings decide how T1j plays. They are part of the measured agent, so
they are frozen here rather than inherited from whatever a constructor leaves
behind:

✅ **THE INVENTORY IS CLOSED, and it is smaller than expected.** Derived by
disassembling the pinned jar: **`mdPieRule` is the ONLY `MatchData` field read
anywhere in `InitialMoves`** — a single `getfield MatchData.mdPieRule` in the
whole class. Board dimensions come from the already-sized `Match` boards
(`getXsize()` / `getYsize()`), and the mover from `nextPlayer` / `currentPlayer`.
**No human flags, names, game-over flag or any other `MatchData` field is
required**, and `secondToFourthMove()` and `fifthOrMoreMove()` read none at all
— which is why §4A's plies 1/3/5 returned moves with `matchData` still null.

| field | value | status |
|---|---|---|
| `mdPieRule` | 🔴 **false** | **OPERATIVE.** H4 has no swap rule; `true` selects a different branch and a different game |
| `mdXsize` | **24** | set and recorded for a coherent object; **not operative here** |
| `mdYsize` | **24** | set and recorded; **not operative here** |
| `mdYstarts` | **true** | set and recorded; **not operative here** |

🔑 **Set the four, record the four, and state plainly that only `mdPieRule` is
operative on this direct-query path.** A coherent object costs nothing and stops
a later reader inferring that the other three were consulted. The reflection
contract is **unchanged at exactly four accesses** on the opt-in path, with
`Match.matchData(write)` added to `AUTHORIZED`.

If implementation finds any further field is required, that is a **design
change** returning here — never a default chosen at the keyboard.

### 2.3 The mechanism: audited private-field injection

Per §1 as corrected, reflection is the **narrowest recommended** route — not the
only conceivable one. The candidate is direct injection of a constructed
`MatchData` into `Match`'s private `matchData` field.

**Constraints:**

* it touches **`matchData` only** — not `moveNr`, `moves`, `boardY`, `boardX`,
  `nextPlayer` or anything else E2 enumerated;
* it is **audited**: the injected object's fields are read back and recorded;
* it runs **only** under the opt-in mode;
* it opens **no new `RightPanel` path**, and the postcondition surface
  (`windows=0`, `frames=0`, `headless=true`, preferences unchanged) must come
  back clean — which is the existing check that would catch it.

⚠ **CORRECTED 2026-09-21.** An earlier draft required that it "never loads
`GuiBoard` or any GUI class." **That is impossible for any non-empty query and
was already false when written.** `E4Preflight` calls `m.setlastMove(...)` to
place each move, and `setlastMove()` routes through the **static**
`GuiBoard.getHoleName()`, so **`GuiBoard` is already resident on every query
with a move in it** — E2 attempt 4 recorded exactly that, and passed.

The real constraint is the one above: **no new `RightPanel` path, and zero
windows and frames.** Residency is permitted; instantiation is not. A
requirement that the qualified harness already violates is not a safeguard — it
is a line nobody could have read and kept.

### 2.4 🔴 It moves the QUERY reflective-access count — a reviewed change

The helper counts its own reflective accesses and the adapter asserts the count:
`QUERY_REFL_N = 3`, `REPLAY_REFL_N = 1` (`e4_screen_integration.py:37-38`),
checked by `check_postcond` on every call.

**The existing three, read from `E4Preflight.java`** — one write and two reads,
exactly as the reviewer characterized them:

| line | access |
|---:|---|
| 125 | `Match.nextPlayer` **(write)** |
| 158 | `FindMove.usealphabeta` (read) |
| 159 | `FindMove.currentMaxPly` (read) |

Adding `Match.matchData` **(write)** makes **four** on the opt-in path —
**expected, and still to be RE-DERIVED from the repaired helper rather than
taken from this table.** Configuration and readback of the injected object
should need no further reflection: the helper is in `net.schwagereit.t1j`, so
`getMatchData()` and its fields are reachable by package access.

Therefore:

* the count is **re-derived, not guessed**, and changed in the same reviewed
  edit as the injection;
* **default callers stay at `refl_n = 3`** — they perform no injection, and
  their count must not drift;
* the opt-in path carries its **own separately pinned count**.

🔑 **`Match.matchData` must also be added to the helper's `AUTHORIZED` list.**
`E4Preflight` asserts `req(reflOk, "only authorized reflective fields used")`
against an allowlist, so an unlisted field fails the postcondition rather than
passing silently. That is the check working, and it must be extended
deliberately.

A repair that quietly relaxed either check instead of updating it would be the
defect class this programme keeps hitting — a guard weakened to fit rather than
re-derived.

---

## 3. CHANGE 2 — `PROC` emission in `E3bDump` (OBSERVATIONAL)

**Independent of Change 1** and reviewable on its own. `E3bDump` emits no `PROC`
line; `E4Preflight` emits exactly one per JVM. §4A measured 0 on ten replays and
1 on ten queries.

* **same schema as `E4Preflight`**: `pid`, `java_version`, `vm`, `headless`,
  `prefs_factory` — so the existing `parse_procs` reads both without a branch;
* **exactly one per JVM**, so the count remains the process count;
* **observational and behaviour-preserving**: it prints, and changes nothing
  about replay.

🔑 **This unblocks the H4 card's §3.2 assertion** — *"replay-process count ==
final ply count + 1"* — which §4A showed is **not satisfiable** as written,
because there is nothing on the replay path to count.

✅ **`REPLAY_REFL_N` STAYS 1.** An earlier draft warned this change "will also
move" it. **It will not**, and the source settles it: `E4Preflight` builds its
PROC line from `ProcessHandle.current().pid()`, three `System.getProperty(...)`
calls and `Preferences.userRoot().getClass().getName()` — **no `reflect()` call
anywhere in it** (`E4Preflight.java:91-95`). Copying that emission into
`E3bDump` adds no reflective access, so the replay count is unchanged at **1**.

That is the difference between the two changes in one line: **Change 1 moves a
count and Change 2 does not**, which is part of why they are reviewable
separately.

---

## 4. THE NEW QUALIFICATION — separately named, NOT a §4A retry

🔴 **§4A is closed.** Its verdict stands on the helper as it was. Because the
helper's behaviour changes, this is a **new qualification with its own name, own
gate, own create-only destination and own card** — not a continuation, and its
results may not be pooled with §4A's.

### 4.0 🔴 TERMINOLOGY: "fallback" is not what the engine does

**`native_low_ply_fallback` is a misnomer and must be retired everywhere.** It
says a search was attempted and fell back. **No search was attempted.**

`FindMove.computeMove()` calls `InitialMoves.initialMove()` **first** and
returns immediately if it gets a move — disassembled: `initialMove` at offset
12, `ifnull 40`, `areturn` at 39. `initialMove` then dispatches on `getMoveNr()`:

| ply | routine | randomized? |
|---:|---|---|
| **0** | `firstMove()` | **yes** — fresh unseeded `Random` (§1.1) |
| **1–3** | `secondToFourthMove()` | **yes** — its own fresh unseeded `Random`, six `nextInt` calls |
| **4–5** | `fifthOrMoreMove()` | **no** — contains no `Random`; may return a native move **or null** |
| **≥6** | — | returns null → **search runs** |

So §4A's `usealphabeta=false, currentMaxPly=0` records at plies 1 and 3 mean
**`secondToFourthMove()` answered**. Nothing searched, nothing fell back.

🔑 **AND THIS RESOLVES A QUESTION THE LOW-PLY CARD LEFT OPEN.** That card
recorded JVM disagreement at plies 1 and 3 (5 of 12 cells) but none at ply 5,
and said in terms: *"What distinguishes ply 5 from plies 1 and 3 is not recorded
here."* It is now. Plies 1/3 are answered by a **randomized** routine; at the
three observed ply-5 positions `fifthOrMoreMove()` returned null so a
**deterministic search** ran instead. `eval_regime` was never the discriminator,
as that card already suspected — **which routine answered** is.

### 4.1 🔴 The matrix must cover plies 1–5, not 0/1/3/5

§4A says in terms that it speaks for **no ply 2 or 4**. Repeating its matrix
therefore **cannot qualify actual H4 play**: T1j Red moves at plies **0, 2, 4**
and T1j Black at **1, 3, 5**, so half of Arm B's early moves would be
unqualified.

**Minimal base matrix: the empty board + plies 1–5 from each of the three
frozen prefix families = 16 positions.** Plies 2 and 4 need **no new input and
no seed**: the frozen families are **already nested** — `o1_center`'s ply-5
prefix `[[11,11],[12,13],[13,12],[10,13],[12,10]]` contains its ply-3 prefix
contains its ply-1 prefix — so plies 2 and 4 are truncations of the ply-5
sequence, still covered by the same pinned hash.

Everything is re-established rather than assumed to have survived: the helper
changed, and a repair that fixed ply 0 while perturbing ply 5 would otherwise go
unseen.

🔑 **Fresh-JVM repetitions for the randomized branches are frozen SEPARATELY,
for OPERABILITY ONLY** — to show the routine keeps returning legal, coherent
moves across processes. They are **not** a probability estimate and the count is
not a sample size (§1.1).

### 4.2 It must establish

1. **multiple fresh-JVM empty-board queries**, to observe the native opening
   distribution **without seeding or replacing** T1j's randomness (§1.1). The
   number of repetitions is frozen in that card, and the observed moves are
   **reported, never constrained to a set** — seven table entries is prior art,
   not an expectation to enforce;
2. **every returned opening move legal and board-coherent** — legal in T1j's
   own report, legal in **our** engine, and coherent with the same-process dump;
3. **four distinguishable paths**, recorded per query — 🔴 **corrected in §4.0 above,
   because "fallback" was never what the engine does**:
   `native_initial_first` · `native_initial_second_to_fourth` ·
   `native_initial_fifth_or_more` · `searched`.
   🔑 §4A deliberately recorded only the neutral `incomplete`, which remains
   correct: it claimed nothing about *why* no search ran;
4. **exactly one `PROC` per query JVM and per replay JVM** — the assertion
   Change 2 exists to make possible;
5. **the injected `MatchData` read back and recorded**, so the frozen settings
   of §2.2 are evidenced rather than asserted.

### 4.3 It stops on

any null or illegal move · any board divergence · any dirty safety surface ·
any lifecycle mismatch. Frozen before any output is seen, as §4A's branches
were.

### 4.4 🔴 Only a clean result authorizes §4B

Not a partial one, and not a repaired-after-the-fact one.

---

## 5. What this card does not do

It writes no code, opens no gate, reserves no seed and runs nothing. It does not
authorize §4B, does not reopen §4A, and does not decide that the repair will be
attempted — only what it would have to be if it is.

**It does not carry `move_count=0` forward** (retired, `10_correction.md` §2),
and the prohibition on a **dummy opening move** stands.

### ✅ The three open questions are ANSWERED — 2026-09-21

**1. Reflection into `Match`'s private state — ACCEPTED, narrowly scoped.**
The harness already depends on qualified private-state access: one
`Match.nextPlayer` write and two `FindMove` search-field reads (§2.4). One
opt-in `Match.matchData` **write** is an incremental, auditable extension of an
existing dependency rather than a new kind of dependency. Conditions, all
binding:

* **default callers remain at `refl_n = 3`**;
* the **opt-in path carries its own separately pinned count** — expected 4,
  re-derived not assumed;
* 🔴 **the result is described as T1j driven through a QUALIFIED PRIVATE-STATE
  ADAPTER, not through its published API.** Every H4 claim inherits that
  wording. E2 was right that this is a materially different claim; the answer
  is to *make* the claim accurately, not to avoid it.

**2. Native opening randomness — IN SCOPE for H4.** The estimand already covers
the **execution/randomization protocol**, not reproducibility from our seed
alone. The realized opening move is observed and persisted; only the latent RNG
state goes unrecorded, and that is a declared property of the protocol. It must
not be seeded, collapsed or required to be diverse (§1.1).

**3. `mdPieRule = false` — CONFIRMED, and required.** Verified in our own
engine, not assumed: `twixt_state.legal_moves()` returns **placement
coordinates only**, `apply_move()` places one peg and switches colour, and
`swap` / `pie_rule` appear **nowhere** in `twixt_state.py`. There is no swap
action and no swap state transition, so a `true` here would have T1j playing a
different game from the one we score.

### What is still owed before implementation

* the **enumeration and justification of any other `MatchData` field** the
  first-move path requires (§2.2) — discovered by reading, in this card, never
  defaulted during implementation;
* the **re-derived** opt-in `refl_n`, and `Match.matchData` added to the
  helper's `AUTHORIZED` allowlist;
* the **new qualification's own card**, with its gate, destination and frozen
  stop conditions.

**Implementation is not authorized by this card.**
