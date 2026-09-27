# H5 IMPLEMENTATION CARD — design only (step 2)

> ⚠ **Read with §10 (amendment, 2026-09-27, after the step-3 build `f80ce42`)** —
> it replaces §2.3 item 2's predecessor rule. The text above §10 is otherwise as
> first committed (`83520b1`).

**Written 2026-09-27. DESIGN ONLY.** No code, no test, no gate, no seed survey,
no registration, no manifest, no JVM, no game, no push.

🔴 **Authority.** The H5 replication card
(`2026-09-25-t1j-h5-replication-card.md`, with its §12 corrections) decides
*what* H5 is; this card decides *how it is built*. Where they disagree, the
replication card wins and the conflict comes back as an amendment. H4's five
bound cards are incorporated by reference where named and are not edited.

---

## 0. The structural decision — reuse H4 by IMPORT, edit NO H4 file

**Measured 2026-09-27**, against H4's committed study manifest
(`docs/superpowers/evidence/t1j-h4-manifests/study_manifest.json`, `b720b28`):

* **36 of the 37** files in H4's bound `code` hash today to exactly the values
  that manifest carries. The one that differs is `e4_screen_reference.py`, changed
  only by H4's closing seed edit (`0aec706`: +22 lines, the five H4 blocks added
  to `EXPOSED_…` and `RETIRED_…`).
* **All five** H4 bound cards hash to the manifest's values.

**Decision.** H5 edits **no H4 file** — no module, card or evidence. H5's modules
**import** H4's play path and call it. That makes §1's "play behaviour unchanged"
a **hash-equality check** (§4.2) rather than a claim, and keeps H4's published
records verifiable against the code that produced them.

**The price, stated.** Two H4 functions read H4 module globals that H5 must
replace — `h4_runner.header_candidate` (the design label, `CARDS`, `CODE`) and
`h4_runner._run_unguarded` (which calls it and names `DESIGNS[mode]`). H5
carries **its own copy of those two functions only**. A test parses both
functions from source and requires their ASTs to be **equal after the named
substitutions of §2.3 and nothing else**, so the copy cannot drift silently.

**Rejected:** parameterizing `h4_runner.py` in place. It would edit a closed,
hashed H4 file, break the 36-file equality above, and leave H4's own tests as
the only evidence that H4's behaviour survived.

---

## 1. Frozen names

```text
runner              scripts/GPU/alphazero/h5_runner.py
qualification       scripts/GPU/alphazero/h5_production_qualification.py
manifest writer     scripts/GPU/alphazero/h5_manifest_writer.py      (outside `code`)
confirmatory        scripts/GPU/alphazero/h5_confirmatory_analysis.py (outside `code`)

gate modules (single declaration each, all created CLOSED, outside `code`):
  scripts/GPU/alphazero/h5_production_qualification_authorization.py
      H5_PRODUCTION_QUALIFICATION_AUTHORIZED = False
  scripts/GPU/alphazero/h5_study_authorization.py
      H5_STUDY_EXECUTION_AUTHORIZED = False
  scripts/GPU/alphazero/h5_aggregation_authorization.py
      H5_STUDY_AGGREGATION_AUTHORIZED = False

tests  tests/test_h5_runner.py, tests/test_h5_production_qualification.py,
       tests/test_h5_manifest_writer.py, tests/test_h5_confirmatory_analysis.py,
       tests/test_gate_inventory.py (updated)
```

* **Gates:** three new. `EXPECTED_GATES` gains them, the count and every
  stale-count word are updated, and a control proves the previous total is stale.
  The total is **derived by `gate_inventory.discover_gates` when step 3 is
  recorded**, not asserted here.
* **Each gate** is read as `MOD.GATE` at call time at **every** public entry
  (function and CLI), **first**, before any file is opened or process started.
  No argv, environment, file or import-hook override.

---

## 2. `h5_runner.py`

### 2.1 Constants

| name | value |
|---|---|
| `DESIGNS` | `{"fixture": "H5_SYNTHETIC_FIXTURE", "study": "H5_STUDY"}` — **no pilot mode** |
| `CARDS` | the replication card, this card, and H4's five bound cards (runner, §4B, replacement, analysis, step-4) — **seven** |
| `CODE` | §2.4 |
| `SEED_BLOCKS` | `{}` — **empty until step 6**; study mode refuses while it is empty |
| `EVIDENCE_DIRS` | `{("study", k): "docs/superpowers/evidence/t1j-h5-study-segment{k}"}`, k = 0–3 |
| pair tag | `h5s{k}`; pair *k* of a block gets `lo + 2k` (Arm A), `lo + 2k + 1` (Arm B) |

Read, never retyped, from `h4_runner`: `DEPTH`, `QUERY_TIMEOUT_S`,
`REPLAY_TIMEOUT_S`, `PLY_CAP`, `ARMS`, `EMPTY_BOARD`, `MANIFEST_BOUND`,
`SEGMENT_DEADLINE_S`, `CLASSIFICATIONS`, `BLIND_FIELDS`, and the exit codes.

### 2.2 Imported from `h4_runner` and called unchanged

`make_schedule`, `check_schedule`, `schedule_digest`, `check_destinations`,
`Results`, `read_records`, `completed_counts`, `split_toolchain`,
`bound_identity`, `binding_differences`, `seed_registry_faults`,
`claim_directory`, `frozen_argmax_config`, `frozen_incumbent_identity`,
`check_incumbent_identity`, `load_production_evaluator`,
`production_incumbent_build`, `_verified_t1j_paths`, `classify`, `play_game`,
`check_game`, `load_games`, `_Stash`, `_trace`, `_stdout_of`, and the exception
types `H4RunError`, `H4RunVoid`. A `run_void` record written by H5 therefore
names `H4RunVoid` — the class that raised, recorded truthfully.

### 2.3 H5's own code — the copy and what is new

* **`header_candidate`** — H5's design label; the bound identity is
  `{**h4_runner.bound_identity(identity, content), "cards": <H5 CARDS hashed>,
  "code": <H5 CODE hashed>}`. `t1j_runtime` is therefore **built by H4's own
  function**, including the key `h4_acceptance: true`, which is **kept by that
  name**: it names the §4B mode, and renaming it would break the equality with
  H4 that §4.2 requires.
* **`_run_unguarded`** — H4's, with exactly these substitutions: `DESIGNS` →
  H5's, `header_candidate` → H5's. The AST test of §0 enforces "nothing else".
* **`canonical_schedule(segment, reference)`** — from `SEED_BLOCKS`, tag `h5s`.
* **`run_study(...)` and `main`** — the public entries. Study mode, like H4's
  `_production`: **no injected seam** of any kind, not even paths; the schedule
  read from the manifest; identity built once and read off the object; canonical
  schedule; registry faults; the entry's `evidence_dir` only; an occupied
  directory refuses; the header bound to the manifest **before** any claim and
  again **after** compiling, before the first game. Fixture mode as H4's:
  negative seeds, stub incumbent and compile, never under the evidence tree.

**New in H5, over H4's runner:**

1. **The manifest must be durable**: tracked at HEAD in this repository and
   unmodified (the H4 analysis card §2.1 rule), checked before it is read.
   H4's runner read its manifest without that check.
2. **Segment order is enforced.** Segment *k* > 0 refuses unless segment
   *k − 1*'s `results.jsonl` is **committed, unmodified, and ends in
   `segment_end`**. This makes the replication card's "commit each segment
   before the next" and "a VOID ends the line" mechanical: after a VOID in
   segment *k − 1*, segment *k* cannot start.
3. **Study mode refuses while `SEED_BLOCKS` is empty**, before anything else
   past the gate.

### 2.4 `CODE` — a rule, with the list derived

`CODE` = every first-party file reachable by import, at any depth, from
`h5_runner.py` and `h5_production_qualification.py`, package `__init__`s
included, **minus the three H5 gate modules**. It is an explicit tuple in the
source; a test recomputes the closure (the H4 closure method of
`tests/test_h4_production_path.py`) and requires equality, and a planted new
import must fail it.

* It **includes** H4's 37 files and any **H4 gate module** the closure reaches
  (e.g. `h4_pilot_authorization.py` via `h4_runner`). Those are hashed on
  purpose: **opening an H4 gate during H5 changes H5's bound `code`** and the
  next H5 action refuses.
* The exact tuple is reported, with its count, in the step-3 commit; the rule
  above is what is frozen.

---

## 3. The two qualifications — ONE module, the ROLE DERIVED

`h5_production_qualification.py` runs H4's 3P-b stages under H5's names —
`incumbent_identity` (design `H5_STUDY`), `checkpoint`, `incumbent_move`
(synthetic seed −1, one empty-board decision, **no T1j call, no game**),
`compile_a`, `compile_b`, `reproducibility`, `study_cli` (the H5 runner CLI,
as a fresh subprocess, exits 5 with its gate shut), `bound_identity` — with the
H4 §12.2.1 durability rules (claim first, stage log fsynced stage by stage, a
STOP keeps every completed observation). It adds **`h4_equivalence`** (§4.2),
and a first stage, **`role`**:

| `SEED_BLOCKS` state at run time | role |
|---|---|
| empty | **EARLY** |
| exactly segments 0–3, 148 seeds each, **every seed** ACCOUNTED and not EXPOSED / RETIRED / TEST_ONLY / CONSUMED | **BINDING** |
| anything else (partial, wrong size, a seed failing the registry) | **STOP** |

🔴 **The role is derived, never chosen.** There is no flag, argument or
environment variable for it.

| | EARLY (step 4) | BINDING (step 7) |
|---|---|---|
| job | find production-path faults before any H5 seed is registered | qualify the **final** code state, after registration |
| record directory | caller-named, create-only, under the evidence tree; **refuses** the binding directory | **only** `docs/superpowers/evidence/t1j-h5-binding-qualification/`; any other refuses |
| binds | **nothing** | the manifests (§5) |
| after it | code may still change (step 6 changes it) | **nothing hashed may change**: H5 `CODE`, the seven cards, the toolchain |

**Why an EARLY record can never bind** — three independent refusals, each with
its own negative control in step 3:

1. the manifest writer refuses any record whose `role` is not `BINDING`;
2. it refuses any record not at the fixed binding path;
3. step 6 edits `e4_screen_reference.py` and `h5_runner.py` (both in `CODE`), so
   an EARLY record's bound `code` cannot equal HEAD — the writer's HEAD
   comparison refuses it too.

The binding directory is create-only and fixed, so **a second BINDING run
refuses**; a new binding qualification (after a STOP or any change) needs an
amendment naming a new directory. **One gate** serves both roles; it is opened
for one run at a time, each under its own authorization.

---

## 4. Pinned H4 inputs

### 4.1 The pins — constants in source, verified before use

| input | path | sha256 | commit | used by |
|---|---|---|---|---|
| **H4 estimate** | `docs/superpowers/evidence/t1j-h4-confirmatory/estimate.json` | `619f515a1535d365160bbcdd470db2b1c7844a5221c8857171305c0556c1f55a` | `3b607aaa6b5b93d95ed68cf3b325609fd7741c27` | confirmatory (§6.3) |
| **H4 study manifest** | `docs/superpowers/evidence/t1j-h4-manifests/study_manifest.json` | `24f40b3f35c605e9bb9a158393a5495901fde86771e3127799d856506d147ff1` | `b720b282dd1653e6275a0051eea6f2506a79ac6e` | qualification (§4.2) |
| **H4 registry, as bound** | `scripts/GPU/alphazero/e4_screen_reference.py` at `b720b28` (`git show`) | `c1021279a2dd37a183c5c7b10577f2a8b541b6714da6c78a6259da4b58228258` | `b720b28…` | qualification (§4.2) |

Each pin is checked **before its content is parsed**: the file (or `git show`
blob) must hash to the pinned value, and a working-tree file must also be
**durable** (tracked at HEAD, unmodified). A mismatch **refuses** — it never
falls back to reading the file anyway. H4's values are **read from these files
by H5 code**, never retyped into H5 source.

### 4.2 `h4_equivalence` — "unchanged" as a check

Run in **both** qualification roles. It **STOPs** unless:

1. every file in the pinned H4 manifest's `code`, **except**
   `e4_screen_reference.py`, hashes today to the manifest's value;
2. `e4_screen_reference.py` at HEAD and at `b720b28` have **equal ASTs once the
   four interval registries** (`ACCOUNTED_…`, `EXPOSED_…`, `RETIRED_…`,
   `TEST_ONLY_SEED_INTERVALS`) **are removed** — so `CONSUMED_SEEDS`, the
   `seed_status` logic and everything else are unchanged;
3. H4's five cards hash to the manifest's `cards` values;
4. H5's bound `t1j_runtime` **equals** the manifest's `t1j_runtime` exactly —
   toolchain content (jar sha256, JDK components, helper sources, class hashes,
   main class), depth, both timeouts, `ply_cap`, `h4_acceptance`;
5. H5's `incumbent_identity` equals the manifest's in **every key except
   `design`**, which is `H4_STUDY` there and `H5_STUDY` here.

All four H4 study segment entries carry the same `code`, `cards`, `t1j_runtime`
and identity (verified 2026-09-27 for all four fields); the check requires that
too and uses entry 0. **A STOP here is the replication card's §1 STOP**: it comes back to
the user as an amendment, never a quiet fix.

---

## 5. `h5_manifest_writer.py` (step 8)

As H4's writer, from the **BINDING** record only (§3), **re-deriving and
comparing** every field: `code` and `cards` at HEAD, committed and unmodified,
equal to the record's `bound_identity` — else something moved after step 7;
`t1j_runtime` from the record, confirmed by a **fresh** toolchain verification;
`incumbent_identity` re-derived; four canonical schedules from `SEED_BLOCKS`,
every seed passing the registry; each `evidence_dir` not yet existing.

One file, `docs/superpowers/evidence/t1j-h5-manifests/study_manifest.json`,
stage `H5_STUDY`, four entries, carrying `qualification_record_sha256` — written
create-only in a create-only directory, **validated by the H5 analysis's own
manifest reader** before it is written. Outside `CODE`; the closure test proves
nothing on the play path imports it.

---

## 6. `h5_confirmatory_analysis.py` (step 13)

### 6.1 Order — refuses unless ALL hold, IN THIS ORDER

1. **`H5_STUDY_AGGREGATION_AUTHORIZED`** is open — checked **first**; an
   input-read spy (`open`, `os.open`, `os.stat`, the hashing path) records **no
   access** with it closed;
2. the destination `docs/superpowers/evidence/t1j-h5-confirmatory/estimate.json`
   is free;
3. **durable**: the H5 study manifest, the four segment files and the pinned H4
   estimate — each tracked at HEAD in this repository, unmodified;
4. the **H4 estimate hashes to its pin** (§4.1);
5. the manifest is an **`H5_STUDY`** manifest with exactly four entries whose
   `cards` and `code` key sets are exactly H5's `CARDS` and `CODE`, the frozen
   runtime scalars, and recomputing digests;
6. **exactly four** segment files, each header bound field by field to **its
   own** entry (design `H5_STUDY`, evidence flag, segment slot, every bound
   field), each ending in `segment_end`, each read by `h4_runner.load_games`,
   each game list equal to its entry's schedule in order;
7. **296 pairs** formed by `h4_confirmatory_analysis.form_pairs` (H4 analysis
   card §3.2, unchanged: two games per pair, one A one B, adjacent, same
   segment, distinct seeds, points recomputed from the frozen winner encoding).

**No pilot artifact exists or is accepted**; the entry takes no parameter for
one. Any H4 file — pilot, segment or manifest — offered as an H5 input fails at
step 5 or 6.

### 6.2 What it computes — the replication card §3–4, and nothing else

| block | content |
|---|---|
| `primary` | mean over all 296 pairs with multiplicity; raw interval via `h4_confirmatory_analysis.interval`; **`verdict`** ∈ `DIRECTIONAL_REPLICATION` / `OPPOSITE_DIRECTION` / `INCONCLUSIVE` by the replication card §3.1 on the **raw computed** bounds (strict: a bound of exactly 0.5 is `INCONCLUSIVE`); its §6 wording; the independence statement **with §12 C1's wording** |
| `cap_free_sensitivity` | whole-pair exclusion, raw interval, `n′ = 0` → "not estimable"; labelled DESCRIPTIVE and CONDITIONAL |
| `per_segment` | in segment order 0–3 (= run order, enforced by §2.3 item 2): mean pair score (n = 74), cap-affected pairs, cap games, `segment_end.total_s`; **no interval, no test, no trend statistic** |
| `concentration` | as H4's confirmatory estimate, via the H4 feasibility module's `blinded_view` and `concentration` plus outcome patterns |
| `h4_beside_h5` | from the **pinned H4 estimate**: H4's `primary` (n, mean, lower, upper), `cap_free_sensitivity`, `caps`; H5's same fields beside them; `difference_of_means` = H5 mean − H4 mean as a number; labelled **DESCRIPTIVE — no interval, no test, no pooled estimate** |
| `inputs` | sha256 of every input, the H4 estimate included |

🔴 **Forbidden in the module** — each a failing test if present: any pooled
mean over H4 and H5 pairs; any interval or test on the difference; any
per-segment interval; any import of an `h3_*` module; any duplicate rule.

H4's **verdict wording is not reused**: `h4_confirmatory_analysis.READINGS` and
`estimate` are not called, because their words are H4's question, not H5's.

### 6.3 Exit and output

As H4 analysis card §4.1 for the confirmatory entry: exit 0 estimate written
(create-only, atomic); 4 refused, nothing written; 5 gate closed, nothing
opened, read or written. Qualified as a **fresh subprocess**.

---

## 7. Seeds — steps 5 and 6

**Step 5, the survey (read-only).** A new script and its output under
`docs/superpowers/evidence/<date>-t1j-h5-seed-survey/`, proposing four 148-seed
candidate blocks, using the H4 v16 provenance method **with one required
correction**:

> 🔴 **Found while writing this card: H4's literal scan cannot see
> underscore-grouped literals.** v16's scanner
> (`01c_collision_proof_v16_provenance.py`) matches
> `(?<![\d.])(2026\d{5})(?!\d)`, so `202_642_000` is invisible to it. A scan on
> 2026-09-27 over the same file set (`scripts/`, `tests/`, `docs/superpowers/`;
> `.py .md .json .jsonl .txt`) found **24 distinct underscore-grouped
> nine-digit `2026…` values in 14 files; 6 lie outside every registered
> interval**: `202630000`, `202630032`, `202699000`, `202699032` (already
> known and avoided in their plain form) and **`202642000`, `202642148`** —
> an injected-defect mutation string in
> `scripts/GPU/alphazero/injected_defect_controls.py` (the planted wrong block
> of "H4 4b: a seed block is not the registered one", added `aee01a8`).
> **Scope:** that finding says nothing against H4's own blocks, whose nearest
> gap to `202642000` (1,852 from segment 3's end) exceeds the 148 floor; it
> says the method had a blind spot, sitting exactly where a next block would
> naturally be proposed.

**The correction:** H5's scanner matches integer literals **with or without
digit-group underscores** (normalized before comparison), and a control plants
an underscore-grouped candidate seed in an unrelated file and requires the
block to be **rejected** — differentially, against the same block without the
plant. Under the per-occurrence rule the `202642000`–`202642148` mutation
literals are prior terms and are avoided.

**H5's own pinned records** — the survey's and proof's own files, exempt from
the literal term by exact path only — are listed **in the survey's committed
output before the proof runs**, so the exemption cannot be widened after a
result is seen.

**Step 6, the proof and registration.** The proof runs against the registries
as they stand and must be CLEAN **before** the four blocks enter
`ACCOUNTED_SEED_INTERVALS` and `h5_runner.SEED_BLOCKS`; it re-runs after
registration with each block excluded **by identity**; both runs are committed
create-only with the registration. 🔴 **If the method fails against the
replication card, work stops and the question comes back** — the method is not
adjusted and re-run on its own authority.

**Use-recording** stays deferred to one closing edit after H5 ends (H4 step-4
card §2.1); until then the claimed, committed evidence directories are the
one-shot record.

---

## 8. What step 3 must prove — WITHOUT GAMES

Fixtures only (negative seeds, `H5_SYNTHETIC_FIXTURE` headers, temporary
directories, stub incumbent and compile), each check driven through the **real
public entry** in the state it will run in, each gate **opened inside the test**
for the positive path:

| area | must reject / must hold |
|---|---|
| gates | each H5 entry refuses with its gate shut, **before any file access** (spy); CLIs exit 5 as fresh subprocesses |
| gate separation | with every **H4** gate opened in-test, every H5 entry still refuses; with every **H5** gate opened, every H4 entry still refuses |
| seams | study mode refuses each injected seam, paths included |
| H4 material | H5 refuses an H4 manifest, an H4 schedule, an H4 seed (RETIRED), an H4 evidence dir, an H4 results file; H4's runner and analysis refuse H5's |
| seeds | study mode refuses with `SEED_BLOCKS` empty; a non-canonical schedule; a seed failing the registry |
| order | segment *k* refuses when *k − 1* is absent, uncommitted, modified, or ends in `run_void` |
| manifest | an uncommitted or modified manifest; each bound field altered in turn |
| one-shot | an occupied evidence dir refuses, one left by a VOID included; a refusal writes nothing |
| copy | the §0 AST test fails on any substitution beyond §2.3 |
| closure | `CODE` equals the recomputed closure; a planted import fails it |
| roles | empty blocks → EARLY; four registered → BINDING; partial or unregistered → STOP; EARLY refuses the binding dir; BINDING refuses any other dir |
| early never binds | the writer refuses an EARLY record — by role, by path, and by HEAD mismatch, **each separately** |
| equivalence | each of §4.2's five conditions broken in turn → STOP; the pins mismatched → refuse |
| analysis | H4 analysis card §4 controls ported to H5 names; the §3.1 verdict at both boundaries and at exactly 0.5; the pinned H4 estimate altered → refuse; the forbidden list of §6.2 |
| multiplicity | *k* identical pairs count *k* times |
| blinding | nothing on stdout, stderr or the trace carries a `BLIND_FIELDS` field |

**The injected-defect harness** gains a control per new check, harvested in a
disposable worktree before commit and drift-checked against related installed
reasons (the playbook's procedure), with a clean baseline. The regression
selection's failure set must equal its recorded baseline.

---

## 9. What this card does not do

It writes no code or test, creates no gate, surveys or registers no seed, writes
no manifest, starts no JVM, plays no game and pushes nothing. It edits no H4
card, module or evidence, and it does not re-run H4's scan: the §7 finding is a
read-only count over the files named there, not a collision proof.

**Next separately authorized action: step 3 — implement this card, all three
H5 gates CLOSED, fixtures only.**

---

## 10. Amendment — 2026-09-27, after review of the step-3 build

**A1 — §2.3 item 2, the predecessor rule, is REPLACED.** It said segment *k* > 0
refuses unless segment *k − 1*'s `results.jsonl` is *"committed, unmodified, and
ends in `segment_end`"*. The build (`f80ce42`) did exactly that, and review found
it too weak: a committed file holding only a header and a `segment_end` passed,
and its positive test treated that as complete. A bare end marker is not a basis
for spending the next segment's seeds.

**The rule now.** Segment *k* > 0 refuses unless segment *k − 1*'s
`results.jsonl` is committed, unmodified, **and structurally complete against its
manifest entry** — entry *k − 1* of the same committed manifest this run reads:

1. the first record is a header with design `H5_STUDY`, `evidence: true`,
   segment *k − 1*, and every bound field equal to entry *k − 1*'s;
2. for each scheduled task, in schedule order: exactly one `game_start` whose
   task id, pair id, arm, seed and game index equal the schedule's; then only
   `ply` / `process` records of THAT task; then exactly one `game_result` of
   THAT task, before the next `game_start` — a duplicate result, a result for
   any other task, or a result before its game's start refuses;
3. no `run_void`, no unknown record type;
4. the last record, and the only one after the final `game_result`, is a
   `segment_end` with `complete: true` and `games_completed` = 148.

**The blind is PROCEDURAL** (as H4 analysis card §1.2). Parsing JSONL necessarily
reads the bytes that carry outcomes; what is guaranteed is that the check **does
not inspect, compute from, or output any outcome value**. Each record is reduced
to a whitelisted view (record type and identities; the header's bound fields;
`segment_end`'s `complete` and `games_completed`) as soon as it is parsed; the
whitelist is disjoint from `h4_runner.BLIND_FIELDS`, and a test over the source
refuses any outcome-field literal in the check. It does **not** call
`load_games`, which reads winners. Full replay validation stays at aggregation
(§6.1 step 6).

**Tests.** The positive case composes a 148-game predecessor from synthetic
fixture games — its envelope matches an H5 schedule, it is labelled synthetic,
and it lives in a temporary repository outside the evidence tree. Negative
controls cover each clause above, including a header-plus-end-marker file.

