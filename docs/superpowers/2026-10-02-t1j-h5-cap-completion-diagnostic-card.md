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

---

## 7. Amendment A2 — 2026-10-02, the Stage-1 recording contract, committed BEFORE the repair (user-decided)

Review found two Stage-1 blockers in `08baeb09`: failure transcripts were discarded (an
exception became `out: ""`), and the one-shot directory was claimed only **after**
compiling, so a compile failure left no record and a second attempt could look like the
first. The frozen set (§1, §6 A1), the 18 items and every expectation are unchanged.

**A2.1 — Claim before compiling.** Stage 1 refuses, writing nothing, only on **pure**
checks: the gate, the evidence directory, the committed frozen set re-derived equal, the
class directory outside the repository and unoccupied, and our own engine meeting every
frozen expectation. It then **claims** `01_stage1/` (create-only; occupied refuses) **before
compiling**. Everything after the claim — compile, binding, every replay — runs inside a
recorded handler:

* an append-only `log.jsonl`, flushed and fsynced after each completed step (the compile's
  content, then each item's observation), so completed observations survive a kill;
* `record.json` (create-only) on **every** exit: `status` COMPLETE, STOP (an exception) or
  INTERRUPTED (KeyboardInterrupt / SystemExit — re-raised after recording); the
  `failing_stage` (`compile`, `binding`, `replay:<item id>`); the exception type and
  message; the binding if it was reached (else `null`, stated); every completed item.

A claimed directory therefore always names its attempt: a compile failure leaves a STOP
record, and any second attempt finds the directory occupied.

**A2.2 — Every observation keeps what the failure carried.** Per item: stdout (the
transcript), stderr, the return code and the exception type; any field that is not
available is recorded **explicitly, with why**:

| case | stdout | stderr | return code |
|---|---|---|---|
| normal return | the helper's stdout | unavailable — `t1j_adapter.replay` does not return it | the exit status |
| `HelperOutputError` (malformed output) | the stdout the exception carries | unavailable | unavailable — raised before the exit status was returned |
| `TimeoutExpired` | the partial stdout it carries, if any | the partial stderr it carries, if any | unavailable — killed at the timeout |
| any other exception | unavailable | unavailable | unavailable |

A compile failure's carried output, if any, is recorded the same way. Capturing stderr on
a normal return would need a change to `t1j_adapter.replay` — a bound file — and is **not**
made. The Stage-2 permission (§6 A1.2) additionally requires `status: COMPLETE`, and every
stored stream file (stdout and stderr) must be committed and hash to its record.

**A2.3 — Synthetic controls:** malformed replay output; a timeout with partial output; a
compile failure (directory claimed, STOP record naming `compile`, a second attempt
refused); an interruption after a completed item (INTERRUPTED record keeping that item,
the log intact, the interrupt re-raised); a non-COMPLETE record refused by the permission.

**A2.4 — Stage 2: OPEN, not implemented by this amendment.** Stage 2 must not run until its
own contract matches A2.1–A2.2: claim before compiling or loading the model; **every**
query's full stdout (and stderr where available) stored and hashed, not only structured
observations; and **every** exit, including an unexpected exception, written as a record
naming the failing step. That implementation needs its own authorization.

---

## 8. Amendment A3 — 2026-10-02, the Stage-2 recording contract, committed BEFORE the implementation (user-decided)

Stage 1 ran once at `05535ffb` and came back COMPLETE, 18/18 AGREE (committed at
`f7890212`). stderr is unavailable on every item, a limitation the user accepted. That
establishes agreement on the tested completions and blocks; it explains no agent's choice.
This amendment closes §7 A2.4 for Stage 2.

Unchanged: the frozen set, the Stage-1 evidence, §3's counts (K = 3 T1j queries, 2
incumbent decisions, synthetic seed −1), §6 A1.1's judging rules, and every frozen adapter,
agent and configuration.

**A3.1 — Permission before anything is compiled or loaded.** Stage 2 refuses, writing
nothing, on **pure** checks only, in this order:

1. the gate;
2. the evidence directory;
3. the committed frozen set, re-derived equal;
4. `02_stage2/` unoccupied;
5. the class directory outside the repository and unoccupied;
6. **the Stage-1 permission** (§6 A1.2, with §7 A2.2's status and stream rules).

The binding that permission is checked against is re-derived **without compiling**:

* the frozen set's sha256;
* the frozen positions' digests;
* the toolchain content **bound by the H5 study manifest**. Stage 1's compile was required
  to equal it, and its record does.

**A3.2 — The claim, then one deadline.** Stage 2 then claims `02_stage2/` (create-only;
occupied refuses) **before** compiling or loading the model.

The frozen 1-hour bound of §3 **starts at the claim**. It covers the compile, the model load
and every call:

* it is checked before each step;
* it is enforced mid-step by the shared whole-run supervisor (`d1_probe._supervisor`,
  SIGALRM). That supervisor's message says "VOID"; Stage 2 records it as a STOP.

The compile runs under this deadline, not under a separate compile deadline. Its content must
again equal the bound toolchain. Known ceiling: SIGALRM cannot interrupt a C call that never
returns.

**A3.3 — The diagnostic boundary.** T1j is queried through the unchanged §4B path:
`T1jAgent` in H4 acceptance mode, depth 6, the frozen query timeout, `classify_reply`. The
only addition is a recording pass-through around `t1j_adapter.query`. It is handed to the
agent through the agent's existing `_query` parameter, as `runtime_requalification` already
does, and it:

* passes the same arguments in;
* returns the same object;
* re-raises the same exception.

No frozen file is edited. Each T1j query records the following:

| case | stdout | stderr | return code |
|---|---|---|---|
| the query returned (whatever the §4B path then decides) | the full stdout | unavailable — `t1j_adapter.query` does not return it | the exit status |
| `HelperOutputError` (malformed output) | the stdout it carries | unavailable | unavailable |
| `TimeoutExpired` | its partial stdout, if any | its partial stderr, if any | unavailable — killed |
| any other exception, interruption included | unavailable | unavailable | unavailable |

Each T1j query also records:

* the §4B observations the context made for that call: PROC, POSTCOND, MATCHDATA, telemetry,
  classification, outcome and the refusing step;
* the chosen move;
* its judgement under §6 A1.1;
* whether it equals the recorded move.

Each incumbent decision records the move, its judgement, whether it equals the recorded move,
the seconds taken and the seed. The decision is made in-process, so no process stream exists;
the record says so rather than leaving the field blank.

**A3.4 — Persistence.** `log.jsonl` is append-only and fsynced. It records, in order:

1. the compile's content;
2. the model-load identity;
3. each call's row, written together with its stream files before the next call starts.

`record.json` (create-only) is written on **every** exit. Its `status` is one of:

* COMPLETE;
* STOP — any exception: a §4B refusal, a timeout, unreadable output, the deadline, an
  unexpected error, or a call that returns no move;
* INTERRUPTED — KeyboardInterrupt or SystemExit, re-raised after recording.

On any exit other than COMPLETE, the record names:

* the `failing_step`: `compile`, `model_load`, `query:<position>#k` or
  `decision:<position>#k`;
* the exception's type and message;
* the failing call's **partial evidence** — its streams and observations, as far as they
  got;
* every completed row.

A STOP ends the stage; no further call is made.

**A3.5 — Completion and interpretation.** COMPLETE requires no failure **and** the exact
call set derived from the frozen set: 3 queries at each T1j position and 2 decisions at each
incumbent position, so **18 T1j queries and 14 incumbent decisions**, with no omission,
duplicate or extra. A run with no failure whose call set is not exact is **INCOMPLETE**.

Episode readings (§3) exist only for a COMPLETE run; otherwise `episodes` is null, and the
record states why. In a COMPLETE run, any failed control makes every episode
**uninterpretable**.

**A3.6 — Synthetic controls:**

* a compile failure;
* a model-load failure;
* a timeout carrying partial output;
* a §4B refusal;
* malformed output;
* an unexpected exception;
* an interruption after a completed call;
* an occupied `02_stage2/`;
* an occupied class directory;
* a run missing a call: INCOMPLETE, not interpreted;
* permission refused before any claim, compile or load;
* the deadline reached during the compile, and before the model load;
* each completed row persisted before the next call;
* the boundary returning THE object and re-raising THE exception;
* a public-entry run in the real state, with the gate open and a synthetic process boundary.

**A3.7 — Scope.** This amendment authorizes the implementation and its synthetic tests
only. It authorizes no compile, model load, agent query, Stage-2 run, game or push. All 21
gates stay closed.

---

## 9. Amendment A4 — 2026-10-02, a hard deadline, INCOMPLETE diagnostics, the loader control; committed BEFORE the implementation (user-decided)

Review of `e91da6ba` found three gaps before execution. §8 A3 stands except where amended
here. The recording pass-through on `_query` is accepted.

**A4.1 — The hard deadline is external.** SIGALRM cannot interrupt a hung native call (a
model load), so §8 A3.2's in-process deadline is no longer the bound. Stage 2 becomes a
**parent** and a **child** process.

* **The parent** (`run_stage2`, the public entry):
  1. runs §8 A3.1's pure refusals, the Stage-1 permission included;
  2. **claims** `02_stage2/` (create-only) by writing `claim.json`, which holds the claim
     time and the sha256 of a single-use nonce that only the child is given;
  3. starts the child **in its own session**;
  4. waits at most **3600 s from the claim, measured by the parent**. At that limit, or if
     the parent itself is interrupted, it SIGKILLs the child's **whole process group** —
     the child, its JVMs and any other subprocess it started. After a normal exit it also
     kills any member of that group still alive, and records that it did.

  Detection needs nothing from the child: no handler of the child's runs, because the parent
  measures and the kernel kills.
* **The child** (`run_stage2_claimed`, reached only through the parent's argv):
  1. reads the gate;
  2. repeats the pure checks;
  3. takes the claim only if `02_stage2/` holds **exactly** `claim.json` and its nonce
     matches. The journal is created next, so a claim is single-use;
  4. runs §8 A3 with its in-process deadline at **3540 s from the claim**. That 60 s margin
     lets a cooperative STOP be recorded before the hard kill.
* **The parent's record, `watchdog.json`** (create-only), is written on **every** exit,
  separately from the child's `record.json`. It holds:
  * the status;
  * the cause;
  * the limit and the elapsed time;
  * the child's pid and exit status, and the signal sent;
  * whether any survivor was killed;
  * the journal's line count and sha256;
  * whether a record exists, and its status.

  The journal and any record are left **exactly as the child wrote them**. The status is:

  | situation | status |
  |---|---|
  | the parent killed the child (limit or interruption) | **TERMINATED**, never COMPLETE, even if the child wrote a COMPLETE record before hanging |
  | the child exited without a readable record | EXITED_WITHOUT_RECORD |
  | the child exited non-zero, with a record that is not INTERRUPTED | EXITED_ABNORMALLY |
  | otherwise | the record's own status |

* **Ceilings:**
  * a process that leaves the group (setsid) escapes the kill; `subprocess.run` does not
    leave it;
  * if the parent itself is SIGKILLed, the child is unsupervised but keeps its in-process
    deadline.

**A4.2 — INCOMPLETE says what is wrong.** A run with no exception whose call set is not
exact records `failure` as:

* `failing_step: completion_check`;
* `exception_type: null` — no exception was raised, and none is invented;
* the **missing**, **duplicated** and **extra** calls, each a (position, agent, query) triple.

**A4.3 — The loader's wiring is a control.** The real `_incumbent_choose` is run with fake
configuration, identity, loader and builder. The control checks:

* it returns `(callable, identity)`;
* the model is loaded **once**, however many decisions follow;
* a red-to-move and a black-to-move position each get the task whose incumbent colour is
  the mover;
* both tasks carry seed **−1**;
* an identity that is not H5's bound one is refused before any load.

**A4.4 — Synthetic controls,** with harmless dummy processes only:

* the watchdog kills a child that has **every signal blocked**, so no Python handler can run,
  and kills that child's own subprocess;
* a child that finishes returns its exit status;
* an interrupted parent kills the group and re-raises;
* through the public entry:
  * a hung child gives TERMINATED, with the journal preserved byte-for-byte and no record;
  * a COMPLETE record written before a hang is still reported TERMINATED;
  * a child that exits without a record is reported as such;
* claim refusals: no claim, a wrong nonce, a consumed claim;
* the real child invocation exits on the closed on-disk gate;
* INCOMPLETE with missing, duplicated and extra calls.

**A4.5 — Scope.** This amendment authorizes the implementation and its synthetic tests
only. It authorizes no compile, model load, agent query, Stage-2 run, game or push. All 21
gates stay closed. The frozen set, the Stage-1 evidence and frozen agent code are unchanged.
