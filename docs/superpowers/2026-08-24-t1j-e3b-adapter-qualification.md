# E3b — T1j adapter qualification: RAN, PASSED

**Date:** 2026-08-24 · **Status:** E3b **RAN and PASSED.** Qualification exit **0**, smoke exit
**0**. Lockstep agreement on every observable at every ply, across all three fixtures.
· **E4 remains unauthorized.**

Basis: `main` @ `624eb96`, clean, local == remote.
Predecessors: [E2 attempt 4](2026-08-24-t1j-e2-attempt4.md) (headless move),
[E3a](2026-08-24-t1j-e3a-determinism.md) (determinism).
Evidence: `evidence/2026-08-24-t1j-e3b/` — observability check, execution transcript with
substantive commands captured as they ran, legality dump, fixtures, qualification and smoke
outputs, both preference fingerprints, and all five sources.

All six prior E1/E2/E3a evidence directories are **byte-unchanged**.

---

## Step 1 — the observability check passed, so E3b proceeded

The authorized first act. **Every invariant E3b must compare is reachable through T1j's own public
or package-visible API**, with no additional reflection and no rule of T1j recreated.

The two at risk were bridges and winner, and both are available:

- **Bridges** — `Board.isBridged(x, y, direction)` is public and reads the engine's own array;
  `bridge[]` has 4 directions, all with negative Δy, so **each bridge is stored exactly once**, and
  the public static `Board.bridgeEnd(x, y, direction)` yields the far endpoint. The complete bridge
  set is therefore enumerable from public calls.
- **Winner** — `Match.checkForGameOver()` is private and pops a `JOptionPane`, so it is unusable.
  But it decides the winner purely by calling the **public** `Board.checkGameOver()` on each board:
  `boardY` for Y (top-down), `boardX` for X (left-right). Calling those two directly uses **T1j's
  own predicate**, not a reimplementation, and never touches the GUI path.

Recorded caveat: `Board.checkGameOver()` calls `eval.evaluateY(...)` and so **mutates evaluator
state**. The search recomputes evaluation on entry, so it is safe as an observable — noted rather
than assumed.

## Step 2 — the mapping was derived, not assumed

**Stage 1, from observed legality maps.** T1j's `pinAllowed` over the empty board gives: **Y
forbidden on x ∈ {0,23}**, **X forbidden on y ∈ {0,23}**, corners forbidden to both. Ours gives red
forbidden on col ∈ {0,23}, black on row ∈ {0,23}. Enumerating all 8 dihedral transforms × 2 player
assignments, **8 of 16 candidates survive**.

**Stage 2, from full lockstep over every fixture.** The 8 narrow to **4**:

```
SURVIVES  identity     Y=red        rejected  T_identity   Y=black   [side to move Y != X]
SURVIVES  flip_x       Y=red        rejected  T_flip_x     Y=black   [side to move Y != X]
SURVIVES  flip_y       Y=red        rejected  T_flip_y     Y=black   [side to move Y != X]
SURVIVES  flip_both    Y=red        rejected  T_flip_both  Y=black   [side to move Y != X]
```

The transposed family dies at **ply 0** on side-to-move, because the harness binds T1j's first
mover (Y) to our first mover (red).

**The four survivors are a symmetry orbit of the board.** TwixT is invariant under those flips, so
they are indistinguishable by state comparison **by construction** — this is a property of the
game, not a weakness of the derivation. Qualification needs **a** correct mapping, not a unique
one, and the card does not claim uniqueness. The canonical representative is
`identity / Y=red`: **T1j (x, y) ↔ our (row, col) = (y, x)**, **T1j Y ↔ our red**, **T1j X ↔ our
black**.

## Step 3 — lockstep comparison, every ply, three fixtures

Both engines advanced from the **same ordered move sequence** — never converted from a bare
`TwixtState` — and compared after **every** ply on pegs, bridges, side to move, ply/history, and
terminal state with winner attribution.

| fixture | plies | our winner | T1j `termY` / `termX` | pegs | bridges | divergences |
|---|---|---|---|---|---|---|
| `crossing` | 7 | none | false / false | 7 | 1 | **0** |
| `red_win` | 26 | **red** | **true** / false | 26 | 12 | **0** |
| `black_win` | 27 | **black** | false / **true** | 27 | 12 | **0** |

- **The crossing fixture is a genuine block.** Our engine confirms the bridge `((5,6),(7,5))` is
  **absent** because it would cross `((5,5),(7,6))` — one bridge formed where two pegs pairs exist —
  and T1j independently reports `bridges=1`.
- **Both win axes agree.** Red (top-down) sets `termY` only; black (left-right) sets `termX` only.
  That is win-axis and winner attribution confirmed for both colours against T1j's own predicate.
- Fixtures are **deterministic scripted sequences**, built and verified against our engine before
  either engine was driven: no agent chose a move, no search ran during a sequence, no seeds.

## Step 4 — the move smoke

One fixed position, fixed ply 3, exactly one `computeMove`. **The returned move was validated in
our engine before being applied anywhere.**

```
T1j returned (x=15, y=15)  usealphabeta=true  currentMaxPly=4
  -> ours (row=15, col=15)
PASS  returned move is not the null sentinel
PASS  alpha-beta completed fixed ply 3
PASS  OUR engine accepts the move as legal      <- validated BEFORE application
PASS  target intersection empty in our engine
  then applied to both sides:
PASS  pegs agree after the move (7 each)
PASS  bridges agree after the move (1 each)
PASS  side to move agrees          PASS  ply agrees (7)
PASS  neither engine reports a terminal state
```

`(15,15)` is the same move E2 attempt 4 and all 25 E3a queries returned.

## The ply cap is entirely external

`ply_cap` is a **keyword-only argument with no default** on both `t1j_replay()` and
`compare_fixture()`; omitting it raises `TypeError`, and the self-test asserts that on both. The
harness applies the same cap to both engines, and **each engine's ply is derived independently** —
T1j's from `Match.getMoveNr()`, ours from `TwixtState.ply` — then compared. Nothing about the cap
is read from or attributed to T1j.

## The comparator self-tests before it qualifies

Eleven cases, all passing; it refuses to qualify unless every injected divergence is rejected:

| control | rejected on |
|---|---|
| peg divergence | `pegs differ (t1j-only ['1,1,Y'])` |
| coordinate divergence | `pegs differ (ours-only ['5,5,Y'], t1j-only ['9,9,Y'])` |
| player divergence | `pegs differ (ours-only ['5,5,Y'], t1j-only ['5,5,X'])` |
| crossing/bridge divergence | `bridges differ (t1j-only ['1,1|2,3|Y'])` |
| history divergence | `ply 3 != 2` |
| side-to-move divergence | `side to move X != Y` |
| terminal divergence | `terminal/winner T1j {'Y': True…} != ours` |
| cap divergence | `ply count 4 != 3` |
| `ply_cap` required (comparator) | `TypeError` |
| `ply_cap` required (replay) | `TypeError` |
| baseline | accepted — the control, since a comparator that rejects everything proves nothing |

The smoke separately rejects the null sentinel `(-1,-1)`, exceptions are caught, and every stage
exits nonzero on failure.

## Safety postconditions

Isolated Preferences active, headless required, zero `Window`/`Frame` at start and end, preference
surfaces unchanged (identified Java plist plus directory inventory — not a hash of every file),
and reflection limited to and audited against the three already-qualified fields, all from a
`finally` path. Nothing installed. T1j's source and JAR untouched — the helper is scratch-compiled
against the JAR and lives only in scratch.

---

## What this establishes, and what it does not

**Established:** under the qualified runtime, an adapter can advance T1j and `TwixtState` in
lockstep from a shared move sequence with **exact agreement on every engine-owned observable at
every ply** — coordinates, player mapping, first mover, edge restrictions, legal cells, automatic
bridges, a genuine blocked crossing, side to move, ply and history, natural terminal state, winner
and win axis for **both** colours — plus one validated cross-engine move at fixed ply 3.

**Not established, and not claimed:**

- **A unique mapping.** Four survive; they are a symmetry orbit and equivalent by construction.
- **Conversion from a bare `TwixtState`.** Both engines are advanced from the same ordered move
  sequence. Nothing here supports reconstructing T1j state from one of our positions alone.
- **Strength, or anything about play.** One move, no game, no opponent, no seeds.
- **Coverage beyond these fixtures.** Three positions and 60 plies total. Agreement here is not
  agreement everywhere — notably no fixture exercised the draw/board-full path, swap, or
  non-24×24 boards.
- **Absolute placement.** The E0 caveat is untouched: T1j is uncalibrated, so even a fully
  qualified adapter yields an **ordering**, not a placement.

## Where the ladder stands

| gate | question | status |
|---|---|---|
| E0 | is there a candidate? | T1j proposed |
| E1 | is the artifact identified? | official-release-qualified |
| E2 | can it be driven headlessly? | **PASSED** |
| E3a | is the move stable? | **PASSED** — 25/25 |
| **E3b** | **do its rules and state match ours?** | **PASSED — 0 divergences** |
| E4 | is it in a usable strength band? | **unauthorized** |

The adapter is qualified, so an E4 endpoint screen would now be measuring strength rather than
debugging plumbing. That screen remains unauthorized, and the shape you set for it stands: weakest
and strongest practical settings, stopping early if either side is obviously saturated.
