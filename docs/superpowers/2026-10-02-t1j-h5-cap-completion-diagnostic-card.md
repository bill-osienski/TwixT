# H5 CAP-COMPLETION DIAGNOSTIC — EXPLORATORY CARD (design only)

**Written 2026-10-02. DESIGN ONLY.** No code, no gate, no JVM, no model load, no query, no
game, no push. Implementation and each run need their own authorization.

🔴 **EXPLORATORY.** This diagnostic follows lead H1 of the H5 weakness review
(`ba50d2d`, `evidence/2026-10-01-t1j-h5-weakness-review-EXPLORATORY/`). It produces **no
strength estimate and no rescoring**: H5's confirmatory result (`8ca7278`) and every H5
record stand unchanged. Nothing here may be pooled with H4 or H5.

---

## 0. The question, and why the board match is not enough

In 13 Arm-B capped games, by **our** rules engine T1j (red) had an immediate win and played
elsewhere, and on the turn before the incumbent (black) left that win open. During play,
T1j's reported position — pegs, bridges, legal moves, side to move, terminal flags — matched
ours at every queried position. **That does not establish agreement about the position
after the unplayed winning move**, which neither engine was ever asked about. Three
explanations are kept apart, and only the third concerns an agent:

| class | meaning |
|---|---|
| **RECONSTRUCTION_DISAGREEMENT** | the engines disagree on the position itself after a witness move: legality, pegs, the bridges it creates, legal moves, side to move |
| **TERMINAL_DISAGREEMENT** | same pegs and bridges, but they disagree whether the move wins (terminal flags), or on the winning path |
| **AGENT_CHOICE** | both engines agree the move is an immediate win, and the agent, queried unchanged, chooses another move |

🔴 **A semantic disagreement (either of the first two, in any episode or control) stops
every agent-weakness attribution**: Stage 2 does not run, and the result returns for review.
And even AGENT_CHOICE says **what** was chosen, never **why**: no search mechanism (candidate
generation, horizon, scoring of won lines) may be inferred from a chosen move or from
telemetry.

## 1. The frozen set — fixed by this card, before any query

**Source:** the committed H5 segment results (sha256 pinned in the review's
`00_inputs.json`) and the review's `03_mechanical_checks.json`
(sha256 `8ef8576ca2a18526437fcfdf8b0d947b6b402e9cf37f672e4d2e41f3390920db`). A **position** is
the first *n* moves of a named game; its **prefix digest** is sha256 of the canonical JSON
list `[[r,c],…]` (separators `,` `:`). Implementation recomputes every digest below and
refuses on any mismatch.

### 1.1 The episodes — the first missed-win episode of each of the 13 caps

The 13 caps reach only **three distinct episode positions**: their move sequences are
identical up to the first missed win (verified for every member) and diverge later.

| episode | games (members) | T1j's position: first *n* moves, digest | witness W (T1j's immediate win) | T1j's recorded move | incumbent's position: first *n* moves, digest | incumbent's recorded move | saving moves S |
|---|---|---|---|---|---|---|---|
| **E1** | 7: h5s0-21-B, h5s1-14-B, h5s1-27-B, h5s1-32-B, h5s1-54-B, h5s1-65-B, h5s2-19-B | 38, `66ef7606efa05e03…b61b` | (17,9) | (16,4) | 37, `26d7d7ad2abc9be3…16ce` | (1,2) | (16,9), (17,9) |
| **E2** | 5: h5s0-69-B, h5s2-23-B, h5s2-35-B, h5s3-36-B, h5s3-45-B | 56, `3644a590483f504b…0873` | (12,5) | (11,7) | 55, `466a8ecead28b58f…3c06` | (1,4) | (12,5) |
| **E3** | 1: h5s3-32-B | 80, `ce4196689c58b958…d98d` | (17,7) | (14,8) | 79, `98a84694d4c9218a…29aa` | (11,10) | (16,6), (17,7) |

Full digests: E1 `66ef7606efa05e032b0ceba0f04cb2720147b416cdae1710461baf291217b61b` /
`26d7d7ad2abc9be3246a15eff538bb652ab9d945a268911cb849667e09cc16ce`; E2
`3644a590483f504b8cf6041284540adfbc402eb03a93295d51832d0c06e20873` /
`466a8ecead28b58f01a4ee2b66045f3e779c1e6e3b39c37a7510613227ee3c06`; E3
`ce4196689c58b9586e31349053e06ff2fe05e7cd0e186d75e490e1b123aad98d` /
`98a84694d4c9218aaad41e9ac5c63115ff75c0f6255765f0cf717438606929aa`.

The incumbent's preceding turn was MISSED_DEFENCE in all 13. In play T1j chose the same
non-winning move at E1 in 7/7 games and at E2 in 5/5.

### 1.2 Controls — selected by rule, frozen here

**Rule:** for each class, the most frequent distinct H5 trajectory (ties: first replay id),
position = the prefix before its final, winning move; witness = that move. Defence
controls: the first two Arm-B incumbent DEFENDED positions by (frequency, replay id, ply).

| control | class | position (game, first *n* moves), digest | witness / recorded move | expected |
|---|---|---|---|---|
| **CT1** | T1j completes as red, interior hole | h5s0-14-B, 30, `0a0f0e6db54de423…434f` | (2,15) | both engines: win; T1j re-query completes |
| **CT2** | T1j completes as red, edge hole | h5s0-09-B, 58, `7afd27fb538832f6…bdf7` | (23,2) | same |
| **CT3** | T1j completes as black, edge hole (Arm A has no interior completion by T1j) | h5s0-09-A, 39, `5ae39981eadd70d2…82d3` | (9,0) | same |
| **CI1** | incumbent completes as black, interior hole | h5s2-05-B, 41, `b8f2977e636bc7f4…d339` | (13,10) | both engines: win; incumbent re-query completes |
| **CI2** | incumbent completes as red, interior hole | h5s0-15-A, 64, `5aaef1686e29e9c9…209a` | (15,16) | same |
| **CD1** | incumbent blocks a single threat (Arm B) | h5s0-21-B, 41, `ab4b4552f037c138…bbe6` | block (16,9) | both engines: after the block, T1j's threat no longer wins; incumbent re-query blocks |
| **CD2** | incumbent blocks a single threat (Arm B) | h5s0-69-B, 127, `1ce728b6ef11e1ac…4800` | block (12,5) | same |

Full control digests: CT1 `0a0f0e6db54de4233c3e00037263241d58457bd4dd42d9752a50a1ed8cc5434f`,
CT2 `7afd27fb538832f6a4ff720e0699a943bb12e171e5fbb088028268347c97bdf7`,
CT3 `5ae39981eadd70d243855a81a9a16c467820412d4c27fd776452c247b59382d3`,
CI1 `b8f2977e636bc7f46d3b8c79157c58fc6e19359f21359fd469aab03f6c9cd339`,
CI2 `5aaef1686e29e9c906a090fe77a15b61c85aec836b7b240083033a8dedbd209a`,
CD1 `ab4b4552f037c1388a9dcb7f80caebdc8f79ab5950736633294ea96eb80bbbe6`,
CD2 `1ce728b6ef11e1ac6601a6906a942b53e4dc695b65505c308ed601f280784800`.

The implementation writes this set, re-derived and digest-checked, as
`00_frozen_set.json` and **commits it before the first JVM starts**.

## 2. Stage 1 — semantic replay in both rules engines (no agent is queried)

**Ours:** `TwixtState` replay of the prefix; the witness's legality
(`is_valid_placement`), the bridges it creates, `winner()` after it, and the **winning
path** (the bridge-connected component through the witness touching both goal edges).

**T1j's:** `t1j_adapter.replay(prefix + [moves])` under H5's **bound** `t1j_runtime` — jar,
JDK components and helper classes hash-equal to the H5 study manifest's, verified before the
first call (a mismatch stops the run) — each call a fresh JVM, its full stdout and exit code
recorded. T1j's per-ply payload is compared with ours by `compare_state` (pegs, bridges,
legal moves, side to move, terminal flags); the witness's legality is read from T1j's legal
list at the prefix; T1j's winning path is checked by running our path finder on **T1j's
reported** pegs and bridges.

| what is replayed | expected in BOTH engines |
|---|---|
| episode: T1j's position + W | legal; same new bridges; **terminal win for red**; a winning path |
| episode: T1j's position + T1j's recorded move | legal; **not terminal** |
| episode: incumbent's position + each saving move s ≠ W, then W | after s, W is legal but **not a win** |
| episode: incumbent's position + s = W | W is then occupied: **illegal for T1j** in both legal lists |
| CT1–CT3, CI1–CI2: position + witness | terminal win for the mover; a winning path |
| CD1–CD2: position + the recorded block, then the threat hole | the threat hole no longer wins |

**Outcome per item:** AGREE, RECONSTRUCTION_DISAGREEMENT, TERMINAL_DISAGREEMENT, or
UNAVAILABLE (T1j's helper refuses, times out, or cannot replay a terminal position —
recorded with its transcript, and **not** counted as agreement).

**Bound:** E1 4 replays, E2 3, E3 4, controls 7 → **at most 18 T1j replay JVMs**, each under
the frozen replay timeout. **Stop rule:** anything but AGREE on any item → Stage 2 does not
run; the stage-1 record returns for review.

## 3. Stage 2 — the unchanged agents, only after Stage 1 is all AGREE

Stage 2 **derives** its permission from the committed, durable Stage-1 record: it refuses
unless every item there is AGREE. There is no flag for it.

* **T1j** at each T1j position (E1–E3, CT1–CT3), through the same §4B query path as in play
  (`T1jAgent`, injected MatchData, frozen depth 6, frozen query timeout, `classify_reply`),
  with the searching JVM's reconstructed position re-bound before the move is read. **K = 3**
  fresh-JVM queries per position → **18 queries**. Recorded: full stdout, exit code, PROC,
  MATCHDATA, POSTCOND, chosen move, telemetry (completed depth, max ply, elapsed), the
  classification, whether the move ∈ W, whether it equals the recorded move.
* **The incumbent** at each incumbent position (E1–E3, CI1–CI2, CD1–CD2) under H5's frozen
  `incumbent_identity` and argmax config — checkpoint hash verified, one model load — with the
  synthetic seed **−1** (the qualifications' convention; no research seed is drawn, no
  registry change). **2 decisions per position** (determinism check) → **14 decisions**.
  Recorded: chosen move, whether it wins / blocks, whether it equals the recorded move, time.
* **Stop rules:** a refusal by `classify_reply` is a recorded STOP. If any **control** fails
  its expectation (CT not completed, CI not completed, CD not blocked), the query path is
  treated as suspect: episode results are reported **uninterpretable** and return for review.
  Total wall clock is bounded at 1 h; at the bound the run stops and records what completed.

**Per-episode reading (descriptive):** T1j completes in k of 3 queries; the incumbent blocks
in j of 2. *0 of 3* with Stage 1 AGREE is an **AGENT_CHOICE** observation — reproduced.
*3 of 3* means the in-play choice did **not** reproduce on re-query (T1j's moves are known to
vary across JVMs, H4 card §0) — reported as not reproduced, not as a weakness. Mixed is
mixed. None of these says anything about the search mechanism.

## 4. Evidence, gate, limits

* **Evidence:** a create-only directory
  `docs/superpowers/evidence/<date>-t1j-h5-cap-completion-EXPLORATORY/` — `00_frozen_set.json`
  (committed before any JVM), `01_stage1/` (record + every transcript), `02_stage2/` (only if
  Stage 1 agreed), and a report labelled EXPLORATORY throughout. Class files compiled into a
  fresh directory outside the repository.
* **Gate (proposed for implementation):** one new single-declaration module,
  `h5_cap_completion_authorization.py`, `H5_CAP_COMPLETION_AUTHORIZED = False`, read first
  at every public entry; the derived gate count rises from 20 to 21 with its inventory test
  and stale-count control. Opened for **one** run at a time — Stage 1 and Stage 2 are
  separate authorizations — and restored on every exit.
* **Code** is read-only towards every H4/H5 module, card and record; the module and its
  synthetic controls live outside every bound `code` list.
* **Not done here, ever:** no game, no strength estimate, no rescoring of any H5 game, no
  change to T1j, the incumbent, the rules engine or any frozen configuration, no inference
  about T1j's search mechanism, no H4–H5 pooling.

## 5. What it may conclude

That at a named, frozen position the two rules engines **agree or disagree** on legality,
bridges, termination and the winning path; and, only if they agree, what each unchanged
agent **chooses** there, how often, and whether that reproduces play. A disagreement is a
rules-reconstruction lead for review; an AGENT_CHOICE observation is a lead for a separately
carded mechanism study — neither is a finding about strength.

**Next separately authorized action:** implementation (gate closed, synthetic controls),
then Stage 1, then — only if all AGREE — Stage 2.

---

## 6. Amendment A1 — 2026-10-02, committed BEFORE implementation (user-decided)

**Three inputs, thirteen provenance links.** The diagnostic has **three episode inputs**,
not 13 independent situations. Every record keeps all 13 source games as provenance links
with their frequencies (E1 7, E2 5, E3 1); no count is ever reported as 13 situations.

**A1.1 — Each control's threat and acceptable replies are pinned** (this corrects §1.2,
whose CD rows named only the recorded block):

| item | threat (the opponent's immediate wins at the position) | recorded move | acceptable reply |
|---|---|---|---|
| CT1–CT3, CI1–CI2 | — | the recorded winning move | **any** immediate winning move for the mover |
| **CD1** | **(17,9)** | (16,9) | any immediate winning move, or **any** move after which the opponent has **no** immediate win |
| **CD2** | **(13,5)** | (12,5) | same |
| episodes, T1j's turn | — | §1.1 | T1j "completes" = any immediate winning move |
| episodes, incumbent's turn | §1.1 W | §1.1 | "blocks" = any immediate winning move or any move removing every immediate T1j win |

Stage 2 judges every reply by these rules (card §7-style precedence: a win counts as
saving), **never** by equality with the historical move; historical-move equality is
reported separately. The frozen-set builder verifies that each pinned threat is
**exactly** the opponent's set of immediate wins at its position and that each recorded
move satisfies its rule — a wrong coordinate refuses.

**A1.2 — Stage-1 permission is integrity-bound (replaces §3's "every item AGREE").**
Stage 2 refuses unless the Stage-1 record is **committed and unmodified at HEAD** and:

1. its item ids equal **exactly** the frozen expected set — the 18 items of §2: no
   omission, no duplicate, no extra;
2. every item is AGREE;
3. its **binding** equals Stage 2's re-derived values: the sha256 of the committed
   `00_frozen_set.json`, the frozen positions' digests, and the toolchain content (jar,
   JDK components, helper class hashes);
4. every transcript file named in the record exists, is committed and unmodified, and
   hashes to its recorded sha256.

"Every present item is AGREE" alone never suffices.

**A1.3 — Path checking.** After a witness, the winning path is validated **on each
engine's reported graph**: on ours from our state, and on T1j's from **T1j's reported pegs
and bridges** — our independent validation of T1j's reported board; T1j does not report a
path. **Different valid paths are not a disagreement.** AGREE requires the same reported
position (`compare_state`), matching terminal recognition, and a valid winning path on each
engine's graph exactly when the move wins (none when it does not).

**A1.4 — Negative controls** (synthetic, in the implementation's tests): wrong threat
coordinates refused; alternative valid winning moves and alternative valid blocking moves
**accepted**; missing, duplicated and extra Stage-1 items refused; each changed binding
(frozen-set hash, a position digest, toolchain content, a transcript hash, an uncommitted
or modified record) refused.

**A1.5 — Gate.** `h5_cap_completion_authorization.H5_CAP_COMPLETION_AUTHORIZED` is created
**False** — the 21st gate, derived by the inventory. The implementation approval authorizes
no JVM, compilation, model load, query, Stage-1 or Stage-2 execution.
