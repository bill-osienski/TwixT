# H5 INCUMBENT TACTICS — EXACT THREAT HANDLING vs PROVEN-RESULT PROPAGATION — EXPLORATORY CARD (design only)

⚠️ **Amended by §10 A1 (2026-10-03). Read §10 first.** It supersedes §3.1, §3.2, §5, §6 and §8
where they conflict.

**Written 2026-10-03. DESIGN ONLY.** No implementation, model load, decision, gate change,
game or push. The implementation and each run need their own authorization.

🔴 **EXPLORATORY; tactical correctness only.** The fixtures and the corpus establish whether two
exactly specified rules behave as specified, and what each would change. They establish **no**
stronger play, select **no** candidate, and change no H4/H5 result. Any candidate or strength
study needs separate review. Causal statements are hypotheses.

---

## 0. Where this comes from

The controlled study (`97363f7d`, card `2026-10-02-t1j-h5-incumbent-defence-controlled-study-card.md`)
closed with three localized, fresh-agent behaviours at the fixtures:

1. **A broad sweep ending in a coordinate tie** (E1 and E2 at 400 simulations).
2. **An immediate winning reply left unvisited** despite substantial search. At E3 the reply is
   (17,7); it is unvisited in all eight arms, under the returned (11,10), (18,13) or (22,7).
3. **A terminal refutation visited but not decisive.** This happened at E2 under FPU −1.0, where
   (12,5) was visited 137 times under (12,4), and at 1600 simulations, where it was visited 16
   times under (2,21). `mcts.py` has no proven-result logic: `_terminal_value` returns ±1, the
   backup averages it, and the readout selects by visit count.

This card separates the two questions those behaviours raise:

* **M1 — exact one-ply threat handling.**
* **M2 — proven-result propagation.**

They are **never combined** (§3.3).

## 1. The closeout features — read from their original design and evidence

**Sources:**

* `docs/superpowers/specs/2026-05-10-closeout-tail-correction-design.md` (§1, §4, §5, §11, §12)
* `docs/superpowers/decisions/2026-05-19-reverted-closeout-experiments.md`
* `connectivity_diagnostics.compute_goal_completion_state`
* `mcts.py`, `force_root_visits` and `apply_closeout_selection_tiebreak`
* `Replays/160-169_Replay/report_160-169.txt` — the Fix 2 run's analyzer report. It is
  **gitignored**, so it exists only on this machine. The spec's own §13 is an unfilled template.

### 1.1 What they tested

* **Purpose.** Self-play **training**: converting the mover's **own** near-completed connection.
  They had no part in defending against an opponent's threat.
* **The trigger.** A **heuristic**, `total_goal_distance == 1`, computed by
  `compute_goal_completion_state`. It uses the mover's dominant component (a component of at
  least 8 pegs) and a component goal distance searched to depth 3. It is **not** an exact
  one-ply scan.
* **Fix 1 — td=1 root visit forcing.** It gives each heuristic "endpoint completion" candidate
  8 forced root visits (at most 4 candidates) inside the 400-simulation budget. It is
  self-play only, root only, and on the **batched** `search_from_root` path.
* **Fix 2 — a narrow tie-break.** Opt-in. It overrides the argmax toward a closeout candidate
  already in the visit top-5, gated on root value ≥ 0.95.

### 1.2 What happened

**Fix 1** (treatment run 140–148, 1000 games):

| metric | result |
|---|---|
| positions where the trigger fired | 984 |
| forced move selected | 99.2% |
| td=1 `completes_endpoint` | 88.6% → 99.2% |
| endpoint at visit rank > 20 | 11.3% → 0.0% |

* The game-count tail targets were met only **partly**. Delay ≥ 10 plies fell 21 → 14, against a
  target of ≤ 10.
* State-cap after detection **rose** 4 → 6; the spec calls this within Poisson noise.

**Fix 2** was deferred after 140–148, then shipped. Per the 160–169 report (iterations 159–168):

| metric | result |
|---|---|
| eligible positions | 3218 |
| overrides | **92** (2.9%) |
| of which to the endpoint / to the reducer | 54 / 38 |

**Reverted.** A trial relaxing Fix 2's value gate to 0.90 **regressed** the closeout tail and
was reverted; the decision record says not to re-test it without a new diagnostic.

### 1.3 Relation to this card

| | closeout features | M1 / M2 here |
|---|---|---|
| threat detection | heuristic, own completion only | **exact** engine scan, own wins **and** opponent threats |
| mechanism | forced visits; a value-gated tie-break | a readout rule on the unchanged final tree |
| path | self-play, batched `search_from_root` | the incumbent's evaluation path, synchronous `search_with_root` |
| proof | none | M2 |

**Nothing is revived.** The closeout code is neither enabled nor modified; both features stay
off in the frozen configuration. In particular, M1 forces no visits and has no value gate.

## 2. The baseline — frozen

* **The agent:** the unchanged incumbent — `calib020_0001` under `frozen_argmax_config()`
  (`search_with_root`, 400 simulations, `c_puct` 1.5, FPU 0.0, argmax readout).
* **Decisions:** a fresh agent per decision, seed −1, with capture off and on.
* **Harness:** `h5_incumbent_defence` — the pass-through, fingerprint, reader and its
  preservation checks — reused by import, not edited.

**Both mechanisms are readout-level functions of the baseline's final tree**, together with
rules-engine scans of the position. The search, priors, FPU, budget and random streams are
therefore **identical** to the baseline's. A decision's M1 and M2 choices are computed from the
same captured tree, never by a changed agent. The baseline tree fingerprint must still be
reproduced (§6).

## 3. The two mechanisms — separate

### 3.1 M1 — exact one-ply threat handling (readout admissibility)

At the decision root, with the mover to play, the rules engine computes by exact full scan:

| symbol | meaning | computed by |
|---|---|---|
| **W** | the mover's immediate winning moves | `D0.one_ply_threats` |
| **T** | the opponent's immediate winning moves, were it to move | the flipped scan |
| **D** | legal moves after which the opponent has **no** immediate win | — |

D is computed with the exact reduction already verified against the full scan in the weakness
review: the mover's peg can only remove opponent wins, never create them.

The **admissible set A** and the class, in **precedence order**:

| order | class | condition | A |
|---|---|---|---|
| 1 | **OWN_WIN** | W is non-empty — terminal precedence; T is irrelevant | W |
| 2 | **DEFEND** | T non-empty, D non-empty. Every move in D answers **every** threat at once, so multiple threats are covered. | D |
| 3 | **UNDEFENDABLE** | T non-empty, D empty — explicit | all legal moves (no change) |
| 4 | **QUIET** | W and T empty | all legal moves (no change) |

**The M1 choice** is the baseline readout restricted to A: the most-visited move in A, with
ties broken by (row, col). Every member of A is equally valid, so alternative winning moves and
alternative defences are all admissible; the choice among them is the search's.

* **No search evidence.** If no move in A was visited, the choice falls to the (row, col)
  minimum at 0 visits, and the record says **"no search evidence"**.
* **Identity by construction.** The M1 choice equals the baseline's whenever the baseline's
  move is already in A. M1 changes a move **only** to replace a non-admissible one.

### 3.2 M2 — proven-result propagation (readout over proven statuses)

Each node of the final tree gets a status, from the perspective of **the player to move at that
node**:

| status | definition |
|---|---|
| **LOSS** | **Terminal** with a winner: the player to move has lost, since in TwixT the player who just moved wins. **Or non-terminal**, expanded, and **every** legal move has a created child with status WIN. |
| **WIN** | Non-terminal, and **some** child has status LOSS. |
| **UNKNOWN** | Everything else. |

* **A move with no child** — never selected — is **UNKNOWN**, never LOSS. Unvisited moves are
  never proven losing.
* A terminal **draw** (the ply cap, or no legal moves) is **not proven**: its status is UNKNOWN.
  It keeps its unchanged value of 0.
* Statuses are computed **bottom-up over the final tree**, read-only. They do **not** steer the
  search; in-search proof propagation would be a separate mechanism (§3.3).

**The M2 choice at the root**, where the root mover sees each child from the opponent's
perspective, follows the first rule that applies:

1. **A proven win.** If some root child is LOSS, the root is WIN: return the most-visited such
   move, ties by (row, col).
2. **Exclusion.** Otherwise, **exclude** every root move whose child is WIN — those moves are
   proven losing for the root mover. Return the most-visited of the rest, ties by (row, col).
   If that move has 0 visits, record **"no search evidence"**.
3. **Every move loses.** If every legal move has been excluded, the root is LOSS and nothing
   avoids the loss. The M2 choice is the **unchanged baseline readout** over all moves, recorded
   as **ROOT_PROVEN_LOSS**.

**Identity by construction.** The M2 choice equals the baseline's unless the baseline's move is
proven losing, or a proven-winning move exists that the baseline did not return.

### 3.3 Never bundled

* M1 and M2 are computed **independently** from the same baseline tree. No record reports their
  composition.
* **Out of scope here**, each a separate mechanism needing its own card:
  * a search-level M1, e.g. restricting or forcing root candidates;
  * in-search proof propagation, e.g. pruning or value override, as in MCTS-Solver;
  * any combination with FPU or budget changes.

## 4. Fixtures — frozen predictions

The fixtures are the five from the frozen set (sha256 `28c03f9f…`, with the A1.6 pins). Each has
**no own win and exactly one threat**, so the fixtures exercise single-threat DEFEND only.
Multiple threats, undefendable positions and terminal precedence come from §5 and §6.

These predictions are **derived from the committed B-repro trees** (`912b601f`/`97363f7d`), which
B-repro must reproduce. A mismatch is therefore an **implementation defect**, not a finding.

| fixture | class | D | baseline move (visits) | **M1 choice** | **M2 choice** |
|---|---|---|---|---|---|
| E1 | DEFEND | (16,9), (17,9) | (1,2) (1) | **(16,9)**: tied at 1 visit with (17,9), resolved by (row, col) | **(1,2)** — unchanged; no root child proven |
| E2 | DEFEND | (12,5) | (1,4) (1) | **(12,5)** (1 visit) | **(1,4)** — unchanged |
| E3 | DEFEND | (16,6), (17,7) | (11,10) (79) | **(16,6)**: both defences at 0 visits, so "no search evidence" | **(11,10)** — unchanged |
| CD1 | DEFEND | (16,9), (17,9) | (16,9) | (16,9) — identity | (16,9) — identity |
| CD2 | DEFEND | (12,4), (12,5), (13,5) | (12,5) | (12,5) — identity | (12,5) — identity |

**Why M2 changes nothing here.**

* At E1–E3 the opponent's winning reply under the returned move was never visited (R1).
* No fixture has an own immediate win.
* Proving a root child LOSS would need either an own immediate win or a fully expanded
  refutation tree — impossible at these visit counts.

M2 can only act on refutations that the baseline search actually visits.

## 5. Synthetic qualification controls — no model; must pass before any run

The positions are built by hand in the rules engine, in the style of the existing ONE/TWO
fixtures. Trees are hand-built or come from the stub-evaluated real search path.

| control | expected |
|---|---|
| own win **and** a threat | OWN_WIN; A = W. An immediate win is chosen over any defence. |
| two own wins | both admissible; the choice is the most visited |
| one threat, two valid defences | both in D; the choice is the most visited; neither is excluded |
| two threats, one common defence | D = that single move |
| two threats, no common defence | UNDEFENDABLE; the choice is unchanged |
| no threat, no win | QUIET; the choice is unchanged |
| the defence set at 0 visits | the (row, col) minimum, flagged "no search evidence" |
| M2: a visited terminal refutation under the argmax | that move is excluded; the next most visited is chosen |
| M2: all visited moves proven losing, some unvisited | an unvisited move is returned, flagged "no search evidence"; unvisited ≠ LOSS |
| M2: every legal move proven losing | ROOT_PROVEN_LOSS; the choice is unchanged |
| M2: a terminal draw | UNKNOWN; not proven |
| M2: a proven-winning root child | chosen |
| both: the tree fingerprint and both streams | unchanged by computing M1 and M2 |

Each control has a deliberately wrong implementation, injected, that the tests must catch.

## 6. The collateral corpus — frozen before any model load

**Source.** The committed H5 records (592 games), read through `h5_weakness_review.load_and_verify`.
The positions are the **incumbent's decision positions**. Excluded:

* every game that sources a fixture or control: the 13 E-members, plus `h5s0-14-B`, `h5s0-09-B`,
  `h5s0-09-A`, `h5s2-05-B`, `h5s0-15-A`, `h5s0-21-B`, `h5s0-69-B`;
* terminal positions.

**Classification** (§3.1) is by the rules engine only. The **quotas** are: 20 OWN_WIN,
20 DEFEND with a single threat, 20 DEFEND with multiple threats, 20 UNDEFENDABLE, and 20 QUIET —
at most **100 positions**.

**Selection.** At most one position per game per stratum. Sampling uses
`random.Random(20261003)` over the eligible positions, in canonical (game, ply) order. A stratum
with fewer eligible positions than its quota is used **whole and reported**; nothing is
substituted.

**The corpus manifest** records each position's game, ply, prefix digest, class, W, T and D. It
is written once and **committed before any model load** — a freeze barrier, as with the
cap-completion frozen set. The run re-derives it and must find it byte-equal.

**Per position**, one capture-off and capture-on pair at seed −1 under the baseline, recording:

* the baseline move;
* the M1 choice and class;
* the M2 choice and every root child's proven status.

## 7. Interpretation rules — frozen before implementation

**Structural checks.** Any violation is an implementation defect: the run's conclusion is
**refused**.

* **M1** may change a move only when the baseline's move is **not** in A, and only to a member
  of A.
* On QUIET and UNDEFENDABLE positions, M1 must equal the baseline.
* **M2** may change a move only when the baseline's move is proven losing, or when a proven win
  exists.
* M2 never excludes an UNKNOWN move, and never treats an unvisited move as LOSS.
* The fixture predictions of §4 hold.

**Reported, descriptively, per stratum** — with no threshold, and nothing that ranks or selects:

* how often each mechanism acts;
* how often it chooses "with no search evidence";
* how often ROOT_PROVEN_LOSS occurs;
* where both the baseline move and the chosen move were visited, the root-mover value of each.

**Allowed conclusions:**

* whether M1 and M2 behave as specified;
* where each would change the incumbent's move;
* how often a tactical repair would rest on no search evidence.

**Not allowed:** that either rule makes the incumbent stronger, or is safe beyond the corpus. A
candidate or strength study needs separate review; the collateral here is tactical and
descriptive, not strength.

## 8. Recording contract, names, size

**The contract** is the established one (cap-completion card §7–§10; the controlled study's §9 A1
and its repaired conclusion boundary):

* pure refusals first;
* a parent claim with a nonce;
* the child in its own session, under the hard watchdog;
* the child's streams captured to files;
* a journal fsynced per observation;
* a record on **every** exit (COMPLETE / STOP / INTERRUPTED / INCOMPLETE).

In addition:

* **The completion check, the M1/M2 computations and the interpretation all sit inside the
  recorded boundary.**
* **B-repro runs first** and must match the five pinned fingerprints and moves of `912b601f`; any
  mismatch STOPs before the corpus.
* Every decision passes the off/on preservation checks.

**Frozen names:**

| item | name |
|---|---|
| module | `scripts/GPU/alphazero/h5_incumbent_tactics.py` |
| gate (the 24th) | `scripts/GPU/alphazero/h5_incumbent_tactics_authorization.py`, declaring `H5_INCUMBENT_TACTICS_AUTHORIZED = False` |
| tests | `tests/test_h5_incumbent_tactics.py` |
| corpus manifest | `docs/superpowers/evidence/t1j-h5-incumbent-tactics-corpus-EXPLORATORY/corpus.json` (a separate freeze step and its own commit) |
| evidence | `docs/superpowers/evidence/t1j-h5-incumbent-tactics-EXPLORATORY/` |
| CLI | `{freeze-corpus, run, run-claimed}`, with no `--out-dir` |

**Size:**

* **Decisions:** at most 210 — B-repro 10, plus the corpus at most 100 × 2.
* **Model load:** one.
* **Seed:** −1 only, synthetic.
* **Hard deadline:** 1800 s.

**Authorization ladder:**
1. implementation, with the synthetic controls of §5 and the gate closed;
2. the corpus freeze and its commit;
3. one run.

Each step is a separate authorization.

## 9. Scope

This card authorizes **nothing executable**: no implementation, model load, decision, gate
change, game, T1j call or push.

---

## 10. Amendment A1 — 2026-10-03, four corrections before implementation (user-decided)

Review of `417b6efa` kept the separation:

* M1 may act without search evidence;
* M2 uses only proofs present in the unchanged tree;
* neither becomes a candidate.

It corrected four points. The closeout history (§1) stands.

### A1.1 — Corpus selection, executable and unambiguous (supersedes §6's selection rule)

**Universe.** Every incumbent decision position of every committed H5 game, with these
exclusions:

* the fixture and control source games (§6);
* positions at ply ≥ 280 — not decision positions under the protocol (A1.2).

Each position gets:

* **a class**, under protocol semantics (A1.2);
* **a state digest:** the sha256 of a canonical JSON with
  * the sorted pegs, as `[row, col, colour]`;
  * the sorted bridges;
  * the side to move.

**Fixture-equivalent positions.** A position whose state digest equals that of any of the
5 fixture positions, or of the 7 control positions in the frozen set, is **removed from
sampling** and listed separately as `fixture_equivalent`. It is never decided as corpus.

**Grouping and sampling,** per class *s* in the order OWN_WIN, DEFEND_SINGLE, DEFEND_MULTI,
UNDEFENDABLE, QUIET:

1. **Group by game.** For each game *g* with at least one eligible position in *s*, pick **one
   representative**:
   `random.Random(f"20261003:{s}:{g}").choice(sorted plies of g in s)`.
   String seeds are hashed with sha512 by Python, so this is reproducible.
2. **Sample games.** Let G_s be those games, sorted by `task_id`. Take
   `random.Random(f"20261003:{s}").sample(G_s, min(20, len(G_s)))`, and list the result
   sorted by (`task_id`, ply).
3. **Shortage.** If `len(G_s) < 20`, every game in G_s contributes its one representative.
   `quota_shortfall[s] = 20 − len(G_s)` is recorded. Nothing is substituted, and no game
   contributes a second position to the same class.

**Repeated positions.** Different games or classes can yield the same state digest. Such
entries are listed separately as `repeats`, grouped by digest. Each **distinct** state is
decided **once**: the entry first in (class order, `task_id`, ply). Per-class statistics
count distinct states, and report repeat multiplicities beside them, never mixed in.

**The manifest** holds, per entry: game, ply, prefix digest, state digest, class (protocol and
board-only, A1.2), W, T, D, and whether it is a repeat. It also holds `fixture_equivalent`,
`repeats` and `quota_shortfall`. It is written once and committed before any model load.

### A1.2 — Cap semantics, frozen (supersedes §3.1 and §3.2 where they are silent on the cap)

**What the code does, as read:**

* `TwixtState` reconstructions — the defence study's `state_of`, and H4/H5's state factory —
  leave `max_plies_limit = None`.
* The H5 searches therefore ran **board-only continuation**, both in play and in the reproduced
  baseline.
* The **protocol** cap of 280 plies was applied by the **play loop**
  (`e4_screen_runner.py:286–293`): `winner()` is checked first, then `ply >= 280` ends the game
  as a draw.
* `apply_move` checks only cell legality, so it permits placements after a terminal position.

**Decision:**

1. **The search is not altered.** It stays exactly the baseline: board-only continuation,
   uncapped states. B-repro's pinned fingerprints depend on this.
2. **The tactical rules use the capped H5 protocol.** That covers M1's classes and M2's statuses
   and certificates, with `ply_cap = 280` an explicit parameter.

| situation | protocol ruling |
|---|---|
| ply ≥ 280 | not a decision position, and never classified |
| a move made at ply *p* wins | a win **even when *p* + 1 = 280** — the winner is checked before the cap |
| the opponent's reply after the mover's move at *p* | counts only if *p* + 1 < 280, i.e. *p* ≤ 278 |
| any "immediate win" at *p* = 279, after the mover's non-winning move | **void** — the game has already ended as a draw |
| a tree node at ply ≥ 280 with no winner | a protocol-terminal **draw**: UNKNOWN, never proven |
| tree nodes beyond such a node | **ignored**: no status or certificate is derived from them |

* **Recorded both ways.** For every position, the board-only class and statuses are recorded
  beside the protocol ones. Any position where they differ is **cap-sensitive** and is listed
  separately.
* **The fixtures** (plies 37–127) are far from the cap, so the §4 predictions are unaffected.

**Synthetic controls,** added to §5. Tests pass the cap as a parameter, e.g. cap = position ply
+ 1 on ONE/TWO-style boards:

| control | expected |
|---|---|
| a win at the boundary | an own win on the last allowed move is OWN_WIN, and its node is a proven terminal LOSS for the next player |
| a terminal draw | a node at the cap with no winner is UNKNOWN; a position at the cap is not classified |
| a purported winning reply after termination | at *p* = cap − 1, the opponent's board-only "immediate win" is void, so the class is not DEFEND. M2 derives nothing from nodes past the cap. The position is flagged cap-sensitive. |

### A1.3 — Reproduction kept separate from corpus preservation (supersedes §8's checks)

| decisions | checks |
|---|---|
| **B-repro** (fixtures) | historical moves **and** the five pinned fingerprints of `912b601f` |
| **corpus** | off/on equivalence; the frozen configuration (no field differs); seed −1; budget 400 |

* Corpus decisions are **never** compared with the in-game move. A different move is data.
* **Computing M1 and M2 is read-only.** For both the off and the on tree, the fingerprint and
  both stream states are recorded before computing M1 and M2, and must be equal after.
  Otherwise: STOP.

### A1.4 — Independently checkable proof witnesses (supersedes §3.2's "statuses computed over the final tree" as sufficient)

**Root statuses alone are not evidence.** Every proven status that M2 reports for a root child
is persisted with a **certificate**. A certificate is reconstructible from the root prefix by
the rules engine alone; it never refers to MCTS objects. The forms:

| form | applies to | content |
|---|---|---|
| `TERMINAL_LOSS` | a node that is protocol-terminal by a **win** | the move path to it; the winner (the player who just moved). Checkable by replay. |
| `WIN` (existential) | a node with a child proven LOSS | `{move, child: <LOSS certificate>}` |
| `ALL_REPLIES_WIN` (universal LOSS) | a non-terminal node where every legal move leads to a proven WIN | `{legal: <the engine's full legal-move list at the node>, children: {move: <WIN certificate>}}`. `legal` must **equal** the engine's list exactly — complete coverage. |

**The verifier**, `verify_certificate(prefix, path, claim, cert, ply_cap)`, is pure and engine-only.
It replays `prefix + path` under protocol semantics and checks the claim recursively.

**When it runs:**

* inside the recorded boundary, for **every** persisted certificate;
* any failure is a **STOP** naming the position and the move.

M2's exclusions and proven-win choices count only when their certificates verify.

**Tests:**

* an internal node stays **UNKNOWN** when one legal child is absent, even if every present child
  is WIN;
* a non-terminal node whose network value or mean is ±1 is **not** proof;
* terminal status comes from the engine's `winner()` on the replayed state, never from `q`,
  `value_sum` or `nn_value`;
* a tampered certificate is rejected. The tampering cases are: a missing legal move, an extra
  move, an illegal move, a wrong winner, and a path that runs past the cap.

### A1.5 — Scope

This amendment authorizes design changes and a local commit only. Implementation, the corpus
freeze, model loading, decisions, gate changes and push all remain unauthorized.
