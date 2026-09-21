# H4 REPAIR DESIGN — `MatchData` initialization and `E3bDump` PROC emission

**Written 2026-09-21. DESIGN ONLY.** Nothing is implemented, no gate exists,
nothing runs, no seed is reserved. This card exists to be argued with.

**Preserves H4.** This is not a retreat from the empty-board study; it is the
separately-reviewed repair that §4A's `STOP_ZERO_LENGTH_REFUSED` requires before
§4B can be contemplated.

> **What the repair must achieve.** `E4Preflight` must be able to ask T1j for a
> move at board-ply 0 without loading a GUI class, and every JVM on both helper
> paths must identify itself — while every existing caller keeps the behaviour
> it was qualified with.

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
4. Therefore **reflection into private state is the only non-GUI route**, and
   E2 refused to take it unilaterally: *"a headless path that depends on private
   internals is a materially different claim from one that uses the published
   API — so E2 stops rather than picking between them."*

🔑 **This card is the review E2 deferred.** Choosing reflection is a material
change in what our harness claims, and it is being made explicitly, in writing,
rather than slipped in as an implementation detail.

### 1.1 🔴 A consequence for the opening that the design must carry

E2 also recorded, on the way past:

> `InitialMoves.firstMove()` selects from a **seven-entry opening table** with
> an **unseeded `new Random()`**. The opening-book branch is therefore
> **nondeterministic by construction.**

So a repaired ply-0 query does **not** return a fixed move. T1j's opening is
drawn from seven entries by randomness **we neither seed nor observe**. That is
native behaviour and **must not be seeded, replaced or collapsed** — it is part
of the agent H4 measures, and it is real entropy at exactly the ply the
superseded H4 card wrongly assumed had none.

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

| setting | value | why |
|---|---|---|
| board size | **24** | `A.BOARD_N`, matching our engine |
| starting player | **red moves first** | TwixT's first player, and H4 Arm B's premise |
| `mdPieRule` | 🔴 **false** | **H4 has no swap rule.** The field whose dereference threw is the one most likely to be set carelessly, and a `true` here would silently measure a different game |

Any other field the path requires must be enumerated **and justified in this
card** before implementation, never discovered and defaulted during it.

### 2.3 The mechanism: audited private-field injection

Per §1, reflection is the only non-GUI route. The narrow candidate is direct
injection of a constructed `MatchData` into `Match`'s private `matchData` field.

**Constraints:**

* it touches **`matchData` only** — not `moveNr`, `moves`, `boardY`, `boardX`,
  `nextPlayer` or anything else E2 enumerated;
* it is **audited**: the injected object's fields are read back and recorded;
* it runs **only** under the opt-in mode;
* it **never** loads `RightPanel`, `GuiBoard` or any GUI class, and the
  postcondition surface (`windows=0`, `frames=0`, `headless=true`) must still
  come back clean, which is the existing check that would catch it.

### 2.4 🔴 It changes the reflective-access count, and that is a reviewed change

The helper **counts its own reflective accesses** and the adapter asserts the
count: `QUERY_REFL_N = 3`, `REPLAY_REFL_N = 1`
(`e4_screen_integration.py:37-38`), checked by `check_postcond` on every call.

**Injection adds reflective accesses, so that count will move.** It must be:

* **re-derived from the repaired helper, not guessed**;
* changed in the same reviewed edit as the injection;
* **different for the opt-in path than for the default path**, since default
  callers perform no injection and their count must not drift.

A repair that quietly relaxed this check instead of updating it would be the
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

⚠ It will also move `REPLAY_REFL_N` if the identity is gathered reflectively.
Same rule as §2.4: re-derive, don't relax.

---

## 4. THE NEW QUALIFICATION — separately named, NOT a §4A retry

🔴 **§4A is closed.** Its verdict stands on the helper as it was. Because the
helper's behaviour changes, this is a **new qualification with its own name, own
gate, own create-only destination and own card** — not a continuation, and its
results may not be pooled with §4A's.

### 4.1 It repeats the FULL 0/1/3/5 matrix

Not only ply 0. The helper changed, so **every** prior observation is
re-established rather than assumed to have survived. A repair that fixed ply 0
and perturbed ply 5 would otherwise go unseen.

### 4.2 It must establish

1. **multiple fresh-JVM empty-board queries**, to observe the native opening
   distribution **without seeding or replacing** T1j's randomness (§1.1). The
   number of repetitions is frozen in that card, and the observed moves are
   **reported, never constrained to a set** — seven table entries is prior art,
   not an expectation to enforce;
2. **every returned opening move legal and board-coherent** — legal in T1j's
   own report, legal in **our** engine, and coherent with the same-process dump;
3. **three distinguishable paths**, recorded per query:
   `native_opening` (ply 0) · `native_low_ply_fallback` · `searched`.
   🔑 §4A deliberately recorded only the neutral `incomplete` because naming a
   fallback is §4B's job. `native_opening` is different: it names **which engine
   routine answered**, which the repaired helper can observe directly;
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

### Open questions for review

1. Is **reflection into `Match`'s private state** acceptable as a permanent
   feature of the H4 harness? E2 declined to decide this unilaterally and it is
   still the decision being made. A "yes" changes what every H4 result claims:
   an agent reached through private internals, not the published API.
2. Should the repaired opening path be **in scope for H4 at all**, given §1.1 —
   T1j's ply-0 move comes from a seven-entry table by randomness we cannot
   observe, so Arm B's first move is drawn from a distribution we can sample but
   never reproduce.
3. `mdPieRule = false` is proposed on the grounds that **H4 has no swap rule**.
   Confirm, because it is a property of the game being measured.
