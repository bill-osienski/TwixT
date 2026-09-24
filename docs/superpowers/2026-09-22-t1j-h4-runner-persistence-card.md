# H4 RUNNER AND CANONICAL PERSISTENCE — CARD

**Written 2026-09-22. DESIGN ONLY.** No runner code, no analysis edit, no seed,
no game, no gate, no aggregation, no push, no §4B rerun.

🔴 **THIS CARD IS THE AUTHORITY for the H4 game runner and its records.** It is
step 1 of the order agreed after §4B: (1) freeze the runner and persistence
contract — **this card**; (2) implement and qualify the runner **without
games**; (3) build **H4-specific** analysis that retains multiplicity — H3 stays
untouched; (4) reserve a
fresh seed block only once the runnable protocol is frozen; (5) run the
outcome-blinded 16-pair pilot, separately. Each is its own authorization.
Implementation conforms to this card or returns here for an amendment.

⚠ **AMENDED 2026-09-23, before step 2 — five corrections; the first version
(`a16e169`) is quoted, not rewritten away:**

1. **H3 is closed and published, and stays unchanged.** The first version said
   step 3 would *"remove H3's duplicate-pair filtering"* and *"changing it is
   step 3"*. Wrong target: step 3 builds **H4-specific** analysis that retains
   multiplicity (§5).
2. **The durable failure record had no schema.** §6 required *"a durable
   terminal record"* that §3.1 never defined. It is now **`run_void`** (§3.4).
3. **Blinding contradicted itself.** §7 said winners go *"**only** to the results
   file"* while §8's replay export requires `winner`. Now: canonical results and
   **derived** exports may carry winners; **stdout, stderr and the progress trace
   never do** (§7).
4. **"Without a seed" was literally impossible** — `game_start` requires one.
   Step 2 uses **named synthetic fixture seeds**, and every qualification output
   is marked **non-evidence** (§10.1).
5. **"Equal digests ⇔ no divergence" overstated a hash.** Canonical **payload**
   equality is what is equivalent to `compare_state`; SHA-256 equality is its
   **integrity fingerprint** under the standard collision-resistance assumption
   (§4).

⚠ **AMENDED 2026-09-23, during step 2 — a conflict between §2 and §10,
resolved by an explicit decision rather than a quiet test edit:**

* §2 requires the runner to construct `T1jRuntime(h4_acceptance=True)`, but §10
  said *"§4B unchanged: its test file passes as is"* — and that file's structural
  test `test_NO_existing_caller_opts_in` allowlisted only the adapter and the §4B
  runner, so the H4 runner failed it. Both could not hold.
* **`scripts/GPU/alphazero/h4_runner.py` is the SINGLE newly authorized
  production opt-in.** Exactly that file is added to that test's allowlist. The
  test's source walk is not weakened or bypassed: **every other caller is still
  refused**, and its clean-baseline control still proves the walker sees a planted
  opt-in.
* §10's claim is corrected accordingly: **§4B's production behaviour and its
  behavioural tests are unchanged; one structural allowlist is deliberately
  extended.**

⚠ **AMENDED 2026-09-23, after step 3 — the header's provenance, authorized
explicitly.** §3.1's header row said *"the governing card commits"*; the step-2
runner in fact recorded the **sha256 of three cards** and **no Python code**. The
H4 analysis card (§0.1) binds a results file to a manifest carrying the hashes of
**four** cards and **five** modules, so no step-2 header could bind. The header
now records **`cards`** — sha256 of this card, the §4B card, the replacement card
**and the analysis card** — and **`code`** — sha256 of `h4_runner.py`,
`e4_screen_integration.py`, `t1j_adapter.py`, `e4_screen_runner.py` and
`h2_match_rules.py`, read from the files at run time. Nothing else in the record
changes.

⚠ **SEQUENCING AMENDMENT 2026-09-23, after `f36d40d` — DESIGN ONLY.** The
agreed order put seed reservation (step 4) straight after the analysis (step 3).
It cannot: `h4_runner.py` still **refuses pilot and study mode** (§10), so the
code that will play does not exist yet, and a manifest (analysis card §0.1) binds
the **sha256 of that code**. Reserving seeds now would put a code change AFTER the
reservation. A new **step 3P — the production runner path — and its no-game
qualification** now sit between step 3 and step 4. See **§12**.

It rests on: the replacement card (`2026-09-21-t1j-h4-replacement-card.md`;
scientific core §1, pilot §5, persistence §7), **the §4B card**
(`2026-09-22-t1j-h4-4b-acceptance-qualification-card.md`) and §4B's **CLEAN**
run (`ca67786`), which qualified the production H4 adapter this runner must use.

> **What the runner must do.** Play **empty-board, colour-reversed pairs**
> between the frozen argmax incumbent and T1j, through the **§4B-qualified** H4
> adapter and the existing harness loop, and persist **every game completely**:
> every move, who made it, which T1j routine answered, every JVM's identity, a
> **re-derivable** coherence digest, and a transcript digest **recomputed** from
> the moves — so any game can be replayed, audited and exported, and no
> observation is ever dropped or merged.

---

## 1. The game — empty-board colour-reversed pairs

🔴 **Not "standard openings".** There is no opening set, no opening name and no
opening digest. Every game starts from the **empty board** (`opening_bound =
0`), built by the existing `make_state_factory` with an **empty move list**.

| | |
|---|---|
| **pair** | two games sharing a `pair_id`, **adjacent** in the schedule, never split across a segment |
| **Arm A** | incumbent **Red**, moves first; T1j Black (`anchor_colour = "black"`) |
| **Arm B** | T1j **Red**, moves first; incumbent Black (`anchor_colour = "red"`) |
| order within a pair | Arm A then Arm B — frozen, so position in the schedule never depends on an outcome |
| points | win 1 · loss 0 · **cap 0.5**, exactly as `play_task` scores them |

The pair buys balance of the first move and **nothing else** (replacement §1.2):
no common-random-number coupling is claimed, and T1j's randomness cannot be
coupled at all.

---

## 2. The runtime — the §4B-qualified configuration, unchanged

| | frozen value | source, **read, never retyped** |
|---|---|---|
| adapter mode | `T1jRuntime(h4_acceptance=True)`, ONE runtime per run, shared by binder and agents | §4B card §3.1 |
| T1j depth | **6** | `h4_repair_qualification.DEPTH` (= `T1J_MDPLY`) |
| query timeout | **120 s**, finite (`t1j_timeout_s`) | `h4_repair_qualification.PER_CALL_TIMEOUT_S` |
| replay timeout | **120 s** (`T1jRuntime.timeout_s`) | same |
| ply cap | **280 total plies** | `l0_match_rules.PLY_CAP` |
| incumbent | the frozen **argmax** configuration, the object that plays read for identity | `h3_study_runner.frozen_argmax_config` pattern |
| compile | once per run through the verified `d1_probe._default_compile` into a **create-only class directory OUTSIDE the repository** | §4B card §7 |

⚠ **H3 put its class directory INSIDE the evidence tree**
(`results_path + ".t1j_classes"`). H4 does not: compiled classes are outputs of
the toolchain, not evidence, and their hashes already travel in the record.

### 2.1 The JVM lifecycle — the existing production path, asserted per game

* **one `T1jAgent` per T1j colour per game** — `play_task` builds agents on first
  use per task and never across tasks;
* **a fresh query JVM for every T1j move**, and **a fresh replay JVM for every
  binder call** — the empty-board opening bind plus one per applied ply;
* no JVM reused across moves, games, arms or pairs.

**Asserted from the records of every game, not from reading code:**

| assertion | from |
|---|---|
| query records == T1j moves in that game | process records, role `query` |
| replay records == final ply + 1 | process records, role `replay` |
| every process record `outcome = accepted` | §4B observations |
| distinct pids == query + replay records | parsed `PROC` |
| **no pid survives its game** — each recorded pid is checked dead after the game (`os.kill(pid, 0)` → `ProcessLookupError`) | recorded pids; a pid-based check cannot match itself the way a `pgrep` pattern can |

---

## 3. THE CANONICAL RECORD — one results file per segment

The **results JSONL** is the authority. Every viewer and report is **derived
from it and hash-bound to it**. Records are written **create-only** (`O_EXCL`),
**flushed and fsynced one by one**, so a kill leaves a truthful prefix.

### 3.1 Record types

| `record_type` | when | carries |
|---|---|---|
| `header` | once, first | design (`H4_PILOT` / `H4_STUDY`), segment, schedule digest, seed-block identity, **the incumbent identity read off the configuration object that plays**, the **T1j runtime identity** (jar sha256, JDK components, compiled-class and source hashes, depth, both timeouts, cap, `h4_acceptance=true`), **`cards`**: sha256 of the four governing cards (this card, §4B, replacement, analysis), **`code`**: sha256 of the five Python modules that play (amendment above) |
| `game_start` | per game, before its opening bind | `task_id`, `pair_id`, `arm` (`A`/`B`), `incumbent_colour`, `t1j_colour`, `seed`, `game_index` |
| `ply` | per applied ply | `task_id`, `ply`, `mover` (colour), **`actor`** (`incumbent` / `t1j`), **`move` `[row, col]`**, `elapsed_s` (monotonic, around the agent call); for a T1j ply also **`source`** (the routine that answered), `t1j_elapsed_us`, and the **ordinal of its query process record** |
| `process` | per T1j subprocess that returned | the §4B observation **verbatim** — role, ordinal, board ply, return code, parsed `PROC`, postcondition fields, `outcome`, `refused_at`, `reason`, and for queries `source`, `MatchData`, telemetry, move — **plus the two coherence digests of §4** |
| `game_result` | per game, after its last ply | `task_id`, `pair_id`, `arm`, `winner`, `terminal_reason`, `plies`, incumbent and T1j points, game `elapsed_s`, **`transcript_digest` recomputed from this game's `ply` records at write time**, query/replay process counts, distinct pids |
| `segment_end` | once, last, on success | games completed, whether the segment is complete, total and setup timing |
| `run_void` | once, last, on an operational failure — **instead of** `segment_end` | the §3.4 schema |

🔴 **The move list IS the reproducibility record** (replacement §3): T1j's
native randomness is unseeded, so no game can be regenerated from our seed. H3
captured every `ply` record in memory and **persisted none of them** — no H3 game
can ever be replayed. **H4 persists every one.**

### 3.2 Identity, digests and timing

* **pair, arm and player identity** on every `game_start`, `ply` and
  `game_result`, so no record needs its neighbours to say whose it is;
* **`transcript_digest`** = `h2_match_rules.transcript_digest(
  h2_match_rules.transcript(plies, result, opening_bound=0))` — the existing
  frozen definition, applied to the **whole** game. **Recomputed** at write time
  from the `ply` records, and **recomputed again** by every reader: a digest read
  back from its own field is a label;
* **timing**: per-ply `elapsed_s`, per-game `elapsed_s`, and a separate **setup**
  time, because the pilot's projection rule needs setup and gameplay apart
  (replacement §5.4). All on a **monotonic** clock.

### 3.3 🔴 Terminal reasons

`win` and `cap` are the only game outcomes. A **qualification ply budget** is
never used in H4 play. Anything else is not a terminal reason — it is an
operational failure (§6).

### 3.4 🔴 `run_void` — the durable failure record

**Exactly one of `segment_end` or `run_void` ends every results file.** A file
that ends in neither was killed, and says so by its truthful prefix.

| field | content |
|---|---|
| `record_type` | `"run_void"` |
| `stage` | `setup` · `compile` · `game` · `post_game_check` · `deadline` — where it failed |
| `task_id`, `pair_id`, `arm`, `ply` | the game and ply in flight, or `null` when none was |
| `classification` | the §6 category: `timeout` · `adapter_refusal` · `unreadable_output` · `replay_mismatch` · `coherence_mismatch` · `missing_record` · `digest_failure` · `process_count` · `illegal_move` · `leaked_process` · `unexpected` |
| `exception` | the exception **type** raised — the classification is decided by type, never by reading a message (§4B card §8) |
| `reason` | the exception's message |
| `stdout` | the full helper transcript when the exception carries one (`HelperOutputError.stdout`, or an `AbortError`'s chained cause), else `null` |
| `completed` | counts **recomputed from the records already written** at the moment of failure: games completed, and `ply`, `process` and `game_result` records |
| `elapsed_s` | monotonic, since the run started |

It is written **best-effort and never masks the error that caused it** — the
harness `Recorder.emit_terminal` contract. It does not make the partial games a
partial result: **a VOID run is VOID.**

---

## 4. 🔑 THE DURABLE COHERENCE FIELD — improving on §4B's boolean

§4B could record only that `compare_state` **passed**; its record kept no dump,
so coherence **cannot be re-derived**. H4 records it re-derivably.

**`position_digest`** — one canonical function over **exactly the fields
`compare_state` compares**:

```
sha256( canonical JSON of {
  "ply", "next_player", "term_y", "term_x",
  "pegs": sorted, "bridges": sorted, "legal": sorted, "history": [...] } )
```

computed twice per T1j subprocess:

| field | from |
|---|---|
| **`t1j_position_digest`** | T1j's **own parsed dump** — the query JVM's pre-move dump, or the replay's final ply block |
| **`expected_position_digest`** | **our** position, rendered in the same vocabulary (`our_snapshot`, the side to move, our legal set, our winner mapped to `term_y`/`term_x`, the submitted history) |

* 🔑 **What is equivalent, and what is only a fingerprint.** The **canonical
  payload** — the JSON object above — **is equal on both sides exactly when
  `compare_state` finds no divergence**: it holds the same fields, compared the
  same way. That equivalence is **proven by a test, field by field**: a control
  makes each compared field differ in turn and shows both that the payloads
  differ and that `compare_state` reports that field.
* **SHA-256 equality is the payload's integrity fingerprint**, not the
  equivalence itself: equal digests imply equal payloads **under the standard
  collision-resistance assumption for SHA-256**. The card states that assumption
  rather than letting a hash stand in for a proof.
* 🔑 **Re-derivable afterwards.** `expected_position_digest` can be
  **recomputed from the persisted move list alone**, by replaying the moves in our
  engine. So a reader can check both that the two stored digests agree and that
  the expected one is the position the moves actually produce.
* It is an **addition** to the §4B observation, computed from the dump the
  adapter **already parses**. The accept/refuse logic does not change. That is
  proven, not asserted: the §4B test file passes unchanged, and a control shows
  identical accept/refuse decisions with and without the new fields. **§4B is not
  rerun, and its evidence is not altered.**

---

## 5. Multiplicity — identical trajectories are DATA

* **Every valid pair is retained with its original multiplicity.** Identical
  transcripts are never excluded, collapsed, merged or reweighted — **not by the
  runner, not in the records, not in any export.**
* A repeated transcript is a **scientific observation** about probability mass
  under the protocol (replacement §1.3). Only integrity faults are fatal: a
  duplicate `task_id`, a repeated seed, mutually inconsistent records.
* 🔴 **H3 IS CLOSED AND PUBLISHED, AND IS NOT CHANGED.** `h3_study_analysis`
  drops duplicate pairs, and with no opening digest its `pair_identity` would
  degenerate to trajectory identity on H4 records — so **no H3 analysis function
  may be applied to H4 records**, now or later, and the runner calls none.
* **Step 3 builds H4-SPECIFIC analysis**, separately authorized, that retains
  multiplicity by construction — proven with a negative control feeding identical
  pairs and asserting every one is counted. H3's code, tests and records stay
  exactly as published.

---

## 6. FAILURES AND STOP RULES — frozen before any game

**An operational failure VOIDs the run** (replacement §5.4: zero allowed). The
runner detects each, writes the **`run_void`** record (§3.4) with that
classification, and stops:

| operational failure | detected as |
|---|---|
| timeout | `subprocess.TimeoutExpired`, the per-call timeouts, or the run's deadline |
| unexpected helper output | an `AbortError` or an unconverted parser error from the H4 adapter — **in play, any adapter refusal is an operational failure**: §4B established that these positions' replies are acceptable |
| replay mismatch | a binder refusal; or `t1j_position_digest ≠ expected_position_digest` |
| missing record | any §3.1 record absent, or the ply sequence not contiguous `1 … plies` |
| digest failure | a transcript digest that does not recompute |
| process-count mismatch | any §2.1 assertion |
| illegal move | `play_task`'s legality check |
| leaked process | a recorded pid alive after its game |

🔑 **Not operational failures**, because they are the phenomena under study: a
**qualified native-initial reply** and a **game cap**.

**One-shot:** a VOID ends that run — no repair, no retry, no reinterpretation.
The full study's per-segment hard deadline is **14,400 s** (replacement §5.4);
the pilot's deadline, projection and cap rules belong to the pilot's own
authorization.

---

## 7. 🔴 OUTCOME BLINDING — enforced by the runner

The replacement card (§5.3) blinds the **decision-maker**, not the disk. The
runner enforces its half:

* **Where outcomes MAY appear:** the **canonical results file** (integrity needs
  `winner` and points) and artifacts **derived from it after the run** — the §8
  replay exports carry `winner`. These are the disk, which the replacement card
  does not blind; the decision-maker's blind is that nobody computes or reads a
  score before the proceed/stop decision is committed.
* **Where they NEVER appear:** **stdout, stderr and the progress trace** — no
  winner, points, per-arm split or running score. Only counts, elapsed time and
  `terminal_reason` categories (`win` / `cap`), which feasibility needs and which
  carry no competitive outcome. The runner writes no export **during** the run;
  exports are derived afterwards, separately;
* the runner computes **no** score, interval or per-arm summary;
* a test asserts the trace and console output carry none of those fields, **with
  a clean-baseline control proving the check would see one**.

Aggregation stays unauthorized until the pilot's proceed/stop decision is
committed.

---

## 8. Viewer-compatible replay export — derived, hash-bound

For every complete game, a standalone **Replay.html**-compatible JSON, in the
format `h2_replay_export` already writes:

* `moves`: every ply as `{turn, player, row, col, bridges_created: [],
  heuristics: {}, search_score: null, root_top1_share: null}`, turns `1 … plies`
  contiguous — **from the empty board, no opening splice**;
* `winner` (`"draw"` for a cap), `starting_player: "red"`, `meta`: board size 24,
  `reason`, `n_moves`, `task_id`, `pair_id`, `arm`, players by colour (T1j vs
  incumbent), the **recomputed** `transcript_digest`, the **sha256 of the source
  results file**, and an `evidence_note` saying it is derived, not a result.

The exporter **refuses** — never guesses — on a digest that does not recompute, a
move count that disagrees with `plies`, a non-contiguous sequence, or mixed task
ids. It reads records only: it never runs an agent, draws a seed or edits
evidence.

---

## 9. Destinations — create-only

| | |
|---|---|
| evidence directory | `docs/superpowers/evidence/<date>-t1j-h4-<stage>[-segment<k>]/`, one per run |
| results JSONL, progress trace, segment report | each `O_EXCL`, claimed **before** compilation and before any JVM |
| replay exports | a create-only subdirectory, written after the run from the results file |
| class directory | create-only, **outside the repository** (§2) |

An occupied destination is refused before anything runs.

---

## 10. What step 2 must prove — WITHOUT GAMES

Step 2 implements this card and qualifies it **without playing a research game
and without a research seed**. Frozen now so the qualification is not shaped by
the code:

* the **whole play path** driven through the real harness `play_task`, the real
  §4B adapter and the real persistence, stubbed at the **process boundary**
  (`subprocess.run`) and with a **stub incumbent** — never at the adapter's seams;
* a stubbed pair written end to end, then **re-read**: moves replay in our engine
  to the recorded result; every transcript digest recomputes; every
  `expected_position_digest` recomputes from the moves; the viewer export
  validates;
* **negative controls**, each shown to reject: every §6 failure; a missing
  `ply`, `process` or `game_result` record; a digest that does not recompute; a
  coherence-digest mismatch; a leaked pid; a record carrying another game's
  `pair_id`; the outcome-blinding check fed a trace that contains a winner;
* **multiplicity**: two identical stubbed games are both persisted and both
  exported — nothing is merged;
* the canonical-payload ⇔ `compare_state` equivalence (§4), field by field, and
  the `run_void` record (§3.4) for each §6 classification;
* §4B unchanged in production behaviour: every §4B behavioural test passes as
  is; the ONE structural change is `h4_runner.py` added to the opt-in allowlist
  (amendment above) — nothing else in that file changes;
* the gate **created closed**, read at both public entries, and the CLI refused as
  a **fresh subprocess**; the injected-defect harness extended and every anchor
  still matching.

### 10.1 🔴 Synthetic fixtures — named, and unmistakably NOT evidence

* **Synthetic seeds are NEGATIVE integers** (`-1, -2, …`). Every research seed
  block this programme has registered is a positive nine-digit `2026…` range, so
  a synthetic seed can never collide with one, and cannot be mistaken for one.
* They reach **only the stub incumbent**. The real incumbent's construction path
  is never given one, and the runner refuses both crossings: **fixture mode
  refuses any non-negative seed**, and **pilot/study mode refuses any negative
  seed**.
* Every fixture output is marked **non-evidence**: its header's `design` is
  **`H4_SYNTHETIC_FIXTURE`** with `evidence: false`; every derived export's
  `evidence_note` begins **`SYNTHETIC FIXTURE — NOT A GAME`**; and fixture mode
  **refuses any destination under `docs/superpowers/evidence/`** — its outputs
  live in test temporary directories only.
* The reader and exporter treat an `H4_SYNTHETIC_FIXTURE` header as fixture data
  and **refuse to present it as a result**.
* A `run_void` fixture is exercised too, so the failure record is qualified
  before any real run needs it.

### Frozen names

```text
gate         H4_PILOT_EXECUTION_AUTHORIZED   (created CLOSED at step 2; the FIFTEENTH gate)
runner       scripts/GPU/alphazero/h4_runner.py
tests        tests/test_h4_runner.py
```

---

## 11. What this card does not do

It writes no code, opens no gate, reserves no seed, plays no game, edits no
analysis, aggregates nothing, pushes nothing, and neither reruns nor alters §4B.

**The next separately authorized action is step 2**: implement this card and
qualify it without games, with the gate created CLOSED.

---

## 12. SEQUENCING AMENDMENT — the production path BEFORE any seed (design only)

**Written 2026-09-23 after `f36d40d`. DESIGN ONLY:** no code, no test, no gate,
no seed, no manifest, no JVM, no push.

### 12.0 Why the order changes

A manifest binds, field by field, the header the pilot will write — including
**`code`**, the sha256 of the five modules that play (analysis card §0.1). Today
`run_games` refuses `pilot` and `study` mode outright (`h4_runner.py`, the
`if mode != "fixture":` refusal): the production incumbent seam was deferred to
*"the pilot's own authorization"*. Building it changes `h4_runner.py`, so its
hash — and possibly others — moves. **Any manifest written before that change
could never bind the run.** Worse, fixing it afterwards would mean editing code
under an already-reserved seed block.

**The order is therefore:**

| step | what | authorization |
|---|---|---|
| 3 ✅ | analysis modules (`a6dea80`, `f36d40d`) | done |
| **3P-a** | **implement** the production path, gates CLOSED, fixtures only | separate |
| **3P-b** | **qualify** it once, **without a game** (§12.2) | separate |
| 4 | fresh collision-checked seed block + committed manifests (§12.3) | separate |
| 5 | the 16-pair pilot | separate |

🔴 **After step 4, every bound file is FROZEN.** A change to any of the five
modules, any of the four cards, the toolchain or the incumbent after the
manifests are committed means those manifests no longer bind anything: the pilot
may not run under them. Recovery is a separately authorized decision — **never an
edit of a committed manifest or a silent reuse of the reserved seeds.**

### 12.1 Step 3P-a — the production path (implementation; gates stay CLOSED)

1. **The incumbent is constructed ONCE, and its identity read off THAT object**
   (the H3 lesson: same function, two objects, one describing a run that never
   happened). The configuration is the frozen argmax one — the eval config of the
   qualified reference path with `selection_mode = h2_match_rules.SELECTION_MODE` —
   built from the **upstream** sources H3 itself read (`twixtbot_g3_reference`,
   `h2_match_rules`, `h2_match_runner`, `e4_screen_reference`), **not** by importing
   `h3_study_runner`: H3 is closed and would otherwise become an unhashed
   dependency of H4's play path.
2. **The recorded identity** = `h2_match_runner.frozen_incumbent_identity()` +
   `argmax_config` read off the object + the H4 design label, checked
   **type-strictly** against the frozen one **before any process starts**.
3. **The evaluator** is loaded by the qualified loader, and the **checkpoint's
   sha1 is recomputed at load** and compared with the identity's before the first
   game. If the existing loader already does this, a control proves it; if not,
   the runner does it.
4. **Pilot and study mode accept NO injected seam** — no `incumbent_build`,
   `evaluator`, `incumbent_identity` or `_compile` argument. An entry that accepts
   its own seams is not a gate. Fixture mode keeps its stubs and its negative seeds;
   the two crossings (§10.1) stay refused.
5. **The header's `t1j_runtime` binds CONTENT, not location.** `_default_compile`
   returns run-local fields too — `classes_dir` (create-only, NEW every run), the
   toolchain `root`, `source` and `verified` count — which no manifest written in
   advance could equal. The header therefore records exactly the analysis card
   §0.1 content: **jar sha256, JDK components, helper source hashes, compiled
   class hashes, main class**, plus depth, both timeouts, `ply_cap` and
   `h4_acceptance=true`. The run-local fields go to a separate **unbound**
   `t1j_local` header field, so where it ran is recorded but not confused with
   what ran.
6. **The runner binds itself before the first game.** Pilot and study mode take
   the **committed manifest** and refuse — after compiling, before any game or T1j
   JVM — unless the header it is about to write equals its manifest entry field by
   field. A run the analysis would refuse is never played.
7. The pilot's existing gate stays the only door to games. **A new closed gate for
   the qualification run** (§12.2) is proposed, making **seventeen**, because 3P-b
   spawns `javac` from the verified JDK and loads the checkpoint — each earlier
   stage that touched the external toolchain took its own reviewed authorization.

**Tests (fixtures only, as in step 2):** the refusals above, each shown to reject
through the public entry with the relevant gate open in the test; the header
projection (content fields present, run-local fields absent from `t1j_runtime`);
the pre-game manifest binding, including a manifest that differs in each bound
field; the injected-defect harness extended.

### 12.2 Step 3P-b — its no-game qualification (one authorized run)

Run once, on the **real** toolchain and checkpoint, gate opened for that run and
restored after. **No game, no T1j query or replay JVM, no research seed, no
outcome.** It establishes:

| check | how |
|---|---|
| incumbent identity | built by the production path, read off the object, equal to the frozen identity type-strictly |
| checkpoint | loaded by the production path, sha1 recomputed, equal to the identity |
| the incumbent can move | **one** incumbent decision from the empty board through the production builder, under a named synthetic seed; a legal move is required. No T1j call, no second ply, no game — a broken seam is found here, not at pilot game 0 with seeds spent |
| toolchain | `verified_paths` + E4's jar pin + JDK components, as `_default_compile` enforces |
| **compile reproducibility** | compile **twice**, into two fresh create-only directories outside the repository; the class-hash maps must be **identical**. If `javac` is not byte-reproducible, **STOP**: class hashes cannot be bound in advance and this card returns for an amendment |
| the header candidate | built by the **same function** the pilot will use, for the pilot stage — `code` (five module hashes at HEAD), `cards` (four), `t1j_runtime` (content only), `incumbent_identity` |
| refusals | the CLI in pilot mode refuses as a fresh subprocess while the pilot gate is closed |

Result is **CLEAN** or **STOP**, recorded create-only under
`docs/superpowers/evidence/<date>-t1j-h4-production-qualification/`, marked
**not evidence of strength**, and committed. A STOP is a result; nothing is
repaired and rerun without a new authorization.

### 12.3 How the final identities enter the manifests (step 4)

The manifests are written **only after 3P-b is CLEAN and committed**, and every
field is **re-derived, then compared** — never copied on trust:

| manifest field | source | must also equal |
|---|---|---|
| `code` | sha256 of the five modules **at HEAD** (committed, unmodified — analysis card §2.1) | the qualification record's `code` — a mismatch means code moved after qualification: **refuse** |
| `cards` | sha256 of the four cards at HEAD | the qualification record's `cards` — any card amended after 3P-b needs a new qualification record first |
| `t1j_runtime` | the qualification record's content projection + the frozen depth, timeouts, cap | a fresh `verified_paths` + JDK verification at manifest time (no compile needed: the class map was proven reproducible) |
| `incumbent_identity` | re-derived by the production path | the qualification record's identity |
| `schedule`, `seeds`, `schedule_digest` | the **fresh, collision-checked** seed block (registered, accounted before use, disjoint from every earlier block); digest recomputed | — |

The pilot manifest and the four study-segment manifests are written in **one**
step-4 commit, together with the seed registration; the study's seeds are
reserved **before** the pilot is run so the pilot's outcome cannot shape them.
The study manifests therefore also freeze everything above for the study: a
change after the pilot — even one the pilot's findings motivate — needs a new,
separately authorized set.

### 12.4 Open decision — for the reviewer, not taken here

⚠ **The `code` field covers five modules** (analysis card §0.1). It does **not**
cover the incumbent's own play path (`twixtbot_g3_reference.py`,
`e4_screen_reference.py`), the evaluator loader (`e4_screen_command.py`), the
compile and toolchain checks (`d1_probe.py`, `t1j_toolchain.py`) or
`h2_match_runner.py` (the identity source). Those are bound only indirectly — by
the incumbent identity's settings and checkpoint hash, and by the toolchain
content hashes. Whether to widen `code` must be decided **before 3P-a**, because
it changes what the header records and what the manifest binds.

### 12.5 What this amendment does not do

It writes no code or tests, creates or opens no gate, reserves no seed, writes no
manifest, plays no game, runs no JVM and pushes nothing. **The next separately
authorized action is 3P-a** (after the §12.4 decision).
