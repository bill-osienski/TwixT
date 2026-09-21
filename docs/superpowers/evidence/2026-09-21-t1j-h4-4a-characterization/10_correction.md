# CORRECTION to `08_outcome.md` — 2026-09-21

`08_outcome.md` is a create-only record of the run and is **not edited**. This
states what it gets wrong. The **verdict, the observations and the branch
classification are unaffected** — what is corrected is a claim about *novelty*,
and one proposed repair that this run's own evidence retires.

---

## 1. "New information" — WRONG. The mechanism was known statically since 2026-08-24

### What I wrote

> The `getMatchData()` finding is new information that a reviewer may weigh

(`08_outcome.md` §6. Commit `2f8e447`'s message carries the same framing under
"UNANTICIPATED FINDING", and my summary to the user called it a finding "nobody
anticipated".)

### Why it is wrong

**`docs/superpowers/2026-08-24-t1j-e2-blocked.md` §2 identified this exact
mechanism, by static reading, a month before §4A ran:**

> **2. At `moveNr == 0` it throws.** That branch calls `firstMove()`, which
> dereferences `match.getMatchData().mdPieRule` — and `Match()`'s constructor
> sets `matchData = null`. The only methods that set `matchData` are
> `prepareNewMatch()` and `updateMatchData()`, and **both call
> `RightPanel.getInstance()`**.

Same field, same call chain, same null, same cause. It is not new, and calling
it new implies nobody could have expected the ply-0 query to fail — when this
programme's own record says the opposite in as many words.

### The correct statement

> **§4A supplies the FIRST RUNTIME CONFIRMATION of a mechanism identified
> statically on 2026-08-24, on the exact path H4 would use** — an `E4Preflight`
> query at board-ply 0 through the qualified adapter, under the frozen
> fresh-JVM lifecycle.

That is still worth having: a static reading says what the code *would* do, and
the E2 report itself was corrected once for overstating a related claim. But
runtime confirmation of a predicted failure is a different and smaller thing
than discovering one.

🔑 **The habit that produced this is the one already on file.** Before calling
anything new, search the programme's own records for it. I did not, on a
question this programme had already answered in writing.

---

## 2. `move_count=0` is RETIRED for this failure — this run's own evidence kills it

### What the card proposed

The H4 replacement card's §4A.2 offers, for an ambiguous zero-length grammar:

> prefer an **explicit count-bearing CLI representation** such as `move_count=0`

### Why this run retires it

**The grammar was never the problem.** The record shows the empty argument tail
parsed correctly end to end:

* the ply-0 **replay** (`E3bDump`) accepted it — exit 0, one ply block, clean
  safety surface;
* the ply-0 **query** (`E4Preflight`) also parsed it, and **emitted a coherent
  empty-board dump** — 0 pegs, 0 bridges, 528 legal cells, matching our engine —
  **before** throwing inside `FindMove.computeMove`.

A helper that could not read a zero-length position could not have produced that
dump. The failure is downstream of parsing, in move computation.

**So `move_count=0` would repair something that is not broken.** It stays
retired unless some *other* evidence resurrects it. The card's other prohibition
is untouched and still binds: **never insert a dummy opening move.**

---

## 3. What is unchanged

The verdict remains **`STOP_ZERO_LENGTH_REFUSED`**, both fired branches remain
as recorded, and §4A remains **complete, closed, and not an authorization for
§4B**. Arm B remains blocked pending separately reviewed repair.

The **§4A.1 branch consequence is also unchanged** and is worth restating,
because §1 above narrows only the *novelty* claim and not the finding: a
zero-length query refusal stops §4A, and any repair is a separate review. That
is what is happening.

`E3bDump` emitting no `PROC` line is **not** corrected here. No prior record of
it has been found, and it stands as reported.
