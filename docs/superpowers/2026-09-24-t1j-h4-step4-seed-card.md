# H4 STEP 4 — SEED REGISTRATION AND MANIFESTS — CARD

**Written 2026-09-24. DESIGN ONLY.** No seed reserved or registered, no manifest,
no gate opened, no code, no push. The only artifact besides this card is a
**read-only collision survey** (§5).

⚠ **AMENDED 2026-09-24 with the 4b authorization — the §6 decisions, taken; the
first version (`808ae60`) is quoted, not rewritten away:**

1. **Deferred use-recording: ACCEPTED, with one correction to the interim claim.**
   The first version said *"Until the closing edit, the committed evidence
   directories ARE the use record."* Before its evidence is committed, a run's use
   record is **the occupied, fsynced evidence directory itself**. The claim is
   **create-only — `os.mkdir`, never `exist_ok=True`** — made **before any seed
   draw or JVM**, its parent directory fsynced. An occupied directory **refuses,
   including one left by a VOID**. (The step-2 runner claimed with
   `os.makedirs(out_dir, exist_ok=True)`: an existing directory was accepted and
   only its files were create-only. That is replaced in 4b.)
2. **This card is BOUND, as the FIFTH card.** It freezes what the manifests only
   carry the results of — the exact blocks, the arm assignment, the collision
   method and the use-recording rule. Its sha256 enters the runner header's
   `cards` and the manifest checks **before 4c**; the card is **frozen after 4c**
   (any later amendment voids 4c for binding, §3).
3. **Evidence directory names are FIXED, not dated**, because the manifest names
   them before the run: `docs/superpowers/evidence/t1j-h4-pilot/` and
   `docs/superpowers/evidence/t1j-h4-study-segment<k>/`. The first version wrote
   `<date>-t1j-h4-pilot/`, a date nobody knows at manifest time.
4. **The schedule is CANONICAL, not merely inside its block.** The first version's
   4b row said the runner refuses *"any seed outside its entry's block"*. The
   runner rebuilds the entry's schedule from §2's rule — block, pair ids, Arm A /
   Arm B seed assignment, the incumbent reference — and refuses any other; the
   registry checks run on top.
5. **Registration waits for the proof.** Collision proof v16 is run against the
   **pre-registration** registries and must be CLEAN before the five blocks are
   written into `ACCOUNTED_SEED_INTERVALS`; it is re-run after registration with
   each block excluded **by identity** (v15's rule), and both runs are committed
   with the registration.

6. **The repo-literal term excludes THIS reservation's own records — by exact
   path, pinned here.** *(Added 2026-09-24 after v16's pre-registration run came
   back NOT CLEAN, `c9d4f5a`; that run is preserved as
   `02a_collision_proof_v16_pre_run_NOT_CLEAN.txt`.)* The failed rule, §5, said:
   *"every nine-digit `2026…` literal in `scripts/`, `tests/`,
   `docs/superpowers/` outside a registered interval is a prior term"*. The
   survey's own committed output and v16's own output list the candidate blocks,
   so the rule counted the proposal's paperwork as a prior reservation and could
   never pass. Corrected: literals are prior terms from **every file except
   exactly these seven**, and from every other file still — the check that found
   `202630000` and `202699000` is kept:

   ```text
   docs/superpowers/2026-09-24-t1j-h4-step4-seed-card.md
   docs/superpowers/evidence/2026-09-24-t1j-h4-step4-collision-survey/01_collision_survey.py
   docs/superpowers/evidence/2026-09-24-t1j-h4-step4-collision-survey/02_collision_survey_run.txt
   docs/superpowers/evidence/2026-09-24-t1j-h4-seed-registration/01_collision_proof_v16.py
   docs/superpowers/evidence/2026-09-24-t1j-h4-seed-registration/01b_collision_proof_v16_amended.py
   docs/superpowers/evidence/2026-09-24-t1j-h4-seed-registration/02a_collision_proof_v16_pre_run_NOT_CLEAN.txt
   docs/superpowers/evidence/2026-09-24-t1j-h4-seed-registration/02b_collision_proof_v16_pre_run.txt
   ```

   An added negative control copies a candidate seed literal into an **unrelated
   file** and requires the block to be **rejected**, so the exclusion cannot
   widen unseen. The re-run is **create-only** (its output file may not exist) and
   carries all 13 earlier controls plus that one; if it is not CLEAN, nothing is
   registered.

It rests on the runner card (`2026-09-22-t1j-h4-runner-persistence-card.md` §12:
the production path and its qualification) and the analysis card
(`2026-09-23-t1j-h4-analysis-card.md` §0.1: what a manifest binds). 3P-b ran once
and was **CLEAN** (`b49b1de`).

---

## 0. The problem this card settles

The shared seed registry — `ACCOUNTED_SEED_INTERVALS`, beside `EXPOSED_…`,
`RETIRED_…` and `TEST_ONLY_…` — lives in **`e4_screen_reference.py`**, which is
one of the **37 files hashed into `code`** (runner card §12.6.1). So:

1. **Registering the H4 seeds there changes a bound hash.** The CLEAN 3P-b record
   (`b49b1de`) binds the pre-registration state; a manifest written after
   registration could never match it.
2. 🔴 **Found while surveying: recording USE there has the same effect, later.**
   H3 recorded each segment's exposure and retirement by editing that file after
   the segment ran. For H4 that edit, between the pilot and a study segment (or
   between two segments), would change `code` and void every study manifest
   committed at step 4.
3. **The H4 runner checks no registry today.** Pilot and study mode refuse only a
   negative seed; H2 and H3 refused any seed absent from `ACCOUNTED_SEED_INTERVALS`
   and any seed exposed or retired.

## 1. Decision — the SHARED registry, then a second qualification

**H4 seeds are registered in `e4_screen_reference.ACCOUNTED_SEED_INTERVALS`**, the
registry every past and future collision proof imports. That is how the
registration stays visible to future proofs **without anyone having to remember
it**: a separate H4 registry would be one more term each future proof must add
by hand — the failure the v15 proof's docstring records ("it keeps having to be
added because the registries are about MATCH seeds and nothing else knows").

**Rejected:** a declaration-only H4 seed module outside `code` (like the gate
modules). It avoids a second qualification but fragments the registry, and the
shared `seed_status` functions — which future proofs and runners use — could not
see it without editing the hashed file anyway.

**The price, accepted:** the CLEAN record at `b49b1de` stops binding. It remains
true of the state it qualified and stays committed as history; the manifests
bind to a **second** qualification of the **final** code state (§3).

## 2. Accounting — five explicit blocks, one seed per game

H3's lesson (segment 0 retry, `2026-09-18`): one contiguous block whose quarters
were all required unspent let ONE segment's VOID block the other three. H4 uses
**five separate blocks**, each registered as its own interval:

| block | interval | seeds | pairs | games |
|---|---|---|---|---|
| `H4_PILOT` | [202 632 000, 202 632 032) | 32 | 16 | 32 |
| `H4_STUDY_SEGMENT_0` | [202 634 000, 202 634 148) | 148 | 74 | 148 |
| `H4_STUDY_SEGMENT_1` | [202 636 000, 202 636 148) | 148 | 74 | 148 |
| `H4_STUDY_SEGMENT_2` | [202 638 000, 202 638 148) | 148 | 74 | 148 |
| `H4_STUDY_SEGMENT_3` | [202 640 000, 202 640 148) | 148 | 74 | 148 |
| **total** | | **624** | **312** | **624** |

* **Seed → game, frozen:** within a block, pair *k* (0-based, schedule order) gets
  seed `lo + 2k` for **Arm A** and `lo + 2k + 1` for **Arm B**; the schedule is
  `h4_runner.make_schedule` over exactly those pairs, so the manifest's `seeds`
  are exactly `range(lo, hi)` in order. Pair ids: `h4p-00 … h4p-15`,
  `h4s{segment}-00 … h4s{segment}-73`.
* **T1j is unseeded**; the seed reaches only the incumbent (its two generators via
  the imported XOR masks — why the survey checks derived streams too).
* **All five registered together** — the study's seeds are reserved **before** the
  pilot runs, so no pilot outcome can shape them (runner card §12.3).

### 2.1 🔴 Recording USE without touching a hashed file during H4

* **No edit to `e4_screen_reference.py` from registration until H4 closes.** The
  five intervals enter `ACCOUNTED_SEED_INTERVALS` once, in step 4b; exposure and
  retirement of H4 blocks are recorded in **ONE** later edit, after the last H4
  run (or the STOP that ends the line), under its own authorization.
* **One-shot without the registry: each run is bound to its evidence directory.**
  Each manifest entry names its create-only **`evidence_dir`**
  (`docs/superpowers/evidence/t1j-h4-pilot/`,
  `docs/superpowers/evidence/t1j-h4-study-segment<k>/` — fixed names, amendment 3).
  In pilot and study mode the runner **refuses any `out_dir` other than that
  one** (step 4b). The directory is claimed **create-only (`os.mkdir`) and
  fsynced before any seed draw or JVM** and committed after the run, so a second
  run of the same entry always finds it occupied and refuses — the job H3 gave to
  "retired on start". An occupied directory refuses **whatever left it there**,
  a VOID included (amendment 1).
* **Until the closing edit, the occupied evidence directories ARE the use
  record** — fsynced at the claim, committed after the run (amendment 1).
  ~~the committed evidence directories ARE the use record~~ The closing edit derives exposure from them (a seed is EXPOSED iff its
  game started, per the results file; every started block is RETIRED whole;
  a block never started — e.g. the study after a pilot STOP — is RETIRED, not
  exposed), exactly the H3 conventions, applied once.
* **Future collision proofs are not weakened meanwhile:** the five blocks are
  ACCOUNTED from 4b on, so no later reservation can overlap them.

## 3. The revised order (supersedes runner card §12's table from step 4 on)

| step | what | authorization |
|---|---|---|
| **4a** | this card + the read-only survey | **this one** |
| **4b** | **implement, gates CLOSED:** register the five blocks in `ACCOUNTED_SEED_INTERVALS` in a commit whose **collision proof v16** (read-only, run against the registries as they stand, v15's method + §5's additions) is committed beside it; runner: pilot/study refuse any seed not ACCOUNTED, or EXPOSED / RETIRED / TEST_ONLY / CONSUMED, and any seed outside its entry's block; runner: `out_dir` must equal the entry's `evidence_dir`; a **manifest writer** (outside `code`, never imported by the play path) that re-derives every field (§4); tests + injected-defect controls | separate |
| **4c** | **re-qualify** the final code state: ONE 3P-b run (same contract, §12.2/§12.2.1), a NEW create-only record `…/<date>-t1j-h4-production-requalification/`. Nothing hashed may change after this run | separate |
| **4d** | run the writer: pilot + four study manifests, re-derived at HEAD and compared with the **4c** record, committed together | separate |
| **5** | the pilot | separate |

🔴 **Any change to a hashed file or bound card after 4c** — including a card
amendment — voids 4c for binding; 4d refuses (§4) and a new qualification is
needed. So every code and card change H4 still needs is made **in 4b**.

## 4. How the manifests are written (4d)

The writer builds each entry from sources and **compares, never copies**:

| field | derived from | must equal |
|---|---|---|
| `schedule`, `seeds`, `schedule_digest` | the registered block (§2) through `make_schedule`, digest recomputed | seeds == `range(lo, hi)`; every seed ACCOUNTED and not unavailable |
| `code`, `cards` | sha256 of the files **at HEAD**, committed and unmodified | the **4c** record's `bound_identity` — else refuse: something moved after 4c |
| `t1j_runtime` | the 4c record's toolchain content + the frozen scalars | a fresh `verified_paths` + JDK verification at write time |
| `incumbent_identity` | re-derived by the production path (design label per stage) | the 4c record's (pilot) / the same with `H4_STUDY` (study) |
| `evidence_dir` | the stage and segment | create-only destination not yet existing |

Five entries, one create-only commit; the writer refuses if any destination or
manifest path exists.

## 5. The read-only collision survey (done under this authorization)

`docs/superpowers/evidence/2026-09-24-t1j-h4-step4-collision-survey/` — the
script and its run. **It is a survey, not the proof**: 4b re-runs the proof
against the registries as they stand then.

* **Method:** collision proof v15 unchanged — each candidate seed and its four
  XOR-mask derivations disjoint from every prior term (all four registries,
  `CONSUMED_SEEDS`, the named spent H1/H2/H3/D1 blocks, the four H3 generation
  ranges no registry holds); derivations injective; nearest other interval at
  least the candidate's own size away. **Added:** the five candidates are checked
  against **each other**, and every nine-digit `2026…` literal in `scripts/`,
  `tests/`, `docs/superpowers/` outside a registered interval is a prior term
  (**except in the seven pinned files of amendment 6**)
  (68 found, mostly exclusive interval ends; notably `202630000` in a test and
  `202699000` in the H4 fixture tests — both avoided).
* **Result:** all five candidates **CLEAN** — no direct or derived-stream
  overlap, injective, nearest gap 1 968 (pilot) and 1 852 (segments) against
  floors of 32 and 148. **Negative controls 12/12 rejected** (spent H3/H2/D1
  blocks, the TEST_ONLY band, both stray literals, a one-seed overlap with segment
  0, a block starting at its end, one seed inside its floor, the live generation
  range).

## 6. Decisions requested before 4b

1. **Deferred use-recording (§2.1)** — one registry edit after H4 closes, with the
   committed evidence directories as the interim record. The alternative (edit
   after each run) would void the study manifests.
2. **Whether this card becomes a BOUND card** (added to `h4_runner.CARDS`, five
   cards). I recommend **not**: it governs how the manifests are made, and the
   manifests themselves are committed and bound field by field. If it is bound,
   the analysis card's four-card rule is amended with it, in 4b.

## 7. What this card does not do

No seed reserved or registered, no manifest, no gate opened, no code or test, no
JVM, no push. The runner card is not edited here; 4b amends its §12 order table
to point to §3 above, as part of the code state 4c qualifies.
