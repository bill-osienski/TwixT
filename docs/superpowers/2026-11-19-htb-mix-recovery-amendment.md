# H-MX recovery amendment 1: one exact replay of the failed fit, with a verified save

**Status:** DRAFT for review (2026-11-19). The amendment and its model-free build are approved; the replay is not.
The build pins this file by sha256. Any change after review is re-pinned and rebuilt before any gate opens.

**Amends:** the H-MX card, `docs/superpowers/2026-11-18-htb-mix-distillation-card.md`, revision 1 (`d2c9b787…`),
which stays frozen and unedited. This amendment adds one stage, between card §14 steps 3 and 4. It changes nothing
else: the mix, the dose, the screens, the export, the match and the pass rule are the card's.

**Links:** ledger Post-closure entry 73 and do-not-repeat #212 (the failed fit). Failed fit record
`docs/superpowers/evidence/htb-mix-fit/record.json`, sha256 `6507d588…10ef9`, committed at `f22a5674`.

## 1. What failed, and what is preserved

The H-MX fit (entry 73) ran all 7,500 updates and applied M1–M6 once: all six held. It then FAILED at the save.
`~/twixt-cache/htb-mix` did not exist, and nothing in H-MX created it. The weights existed only in the process, so
no challenger file exists. The trained-state digest is recorded:
`adba50fb2e241ec0388fa99998b392c3e0b431b55c10676800a546899aee34b2`.

The failed record and its wrapper evidence stay as written. The recovery never writes under
`docs/superpowers/evidence/htb-mix-fit/` or `htb-mix-fit-run/`. It reads the record only after checking its pinned
bytes and its commit. The fit's own entry (`htb_mix run fit`) stays one-shot: its evidence exists, so it refuses.

## 2. The repair

The fault was in the build. These are the repairs:

1. **Prepare the cache directory before training.** The recovery creates `~/twixt-cache/htb-mix` (with any missing
   parents) before any model or label loads. It then proves the directory writable with a create-only probe file,
   which it removes. A failure is a refusal before any work. The challenger path must not already exist.
2. **A verified save.** The save stays H-OP's create-only `save_student`, unmodified. Then:
   1. The file is reloaded strictly with the same loader as the student (`probe_eval.load_network_for_scoring`), in
      evaluation mode.
   2. The reloaded network's state digest must equal `adba50fb…`.
   3. The file's sha256 is recomputed, and must equal the one the save returned.

   **ELIGIBLE is published only after the reload matches.** A mismatch is a STOP with no verdict.
3. **Tests that exercise the real path.** One test drives the real `save_student` and the real loader into a cache
   whose parent directory does not exist, on a full-size network. Another shows that the same save without the
   preparation fails exactly as the run did.

`htb_mix.py` (`79f210cf…`) is **not edited**, so the replay runs the training code that produced `adba50fb…`. The
repair lives in a new module, `htb_mix_recovery.py`.

## 3. The replay: one exact repeat of the failed fit's training

The replay starts from the pinned starting weights and uses everything below exactly as the failed run did. It skips
validation scoring.

| Element | Requirement |
|---|---|
| Starting weights | the current student `0e1cb621…`, state digest `586dca4f…`, loaded twice (reference and student) by the same loader |
| Inputs | both cached label sets through the same pinned records and manifests (`load_sources`, unchanged) |
| Selection | the same overlap and E1–E3 checks. The selection record must equal the failed record's, including both order hashes and the 11 dropped keys |
| Batch order | the same 16 + 16 mixed plan, 7,500 batches (`mixed_plan`, unchanged) |
| Start check | the reference on the first 2,048 mixed rows. The scores must equal the failed record's exactly |
| Training | `fit_mixed`, unchanged: recipe, optimizer, 7,500 updates, CPU, `MLX_ENABLE_TF32=0` |
| Code | `htb_mix.py` sha256 equals the failed record's `module_sha256` (`79f210cf…`). Between the failed launch commit `84ff36b1` and the replay's, no file under `scripts/GPU/alphazero/` changes except the recovery module, the gates and `htb_mix_export.py` (which the fit does not import) |
| Runtime | the main checkout's `.venv` (Python 3.14.10, arm64), MLX 0.30.3 and numpy 2.4.1 (unchanged since 2026-05-09), and the sha256 of `libmlx.dylib` and MLX's core extension, pinned |
| Validation scoring | **skipped.** The replay builds no validation store and scores nothing on either validation set |

**The match that permits a save.** The replay's whole fit record must equal the failed record's `fit` field:
recipe, optimizer, update count, rows per source, all 300 logged losses and gradient norms, surface counts, the
reference digest and **the trained-state digest, which must be exactly `adba50fb…`**. Any difference is a STOP:
"the replay did not reproduce the failed fit". In that case nothing is saved, and no verdict is published.

**Then the save** (§2): prepared directory, create-only save, strict reload, digest `adba50fb…`, file sha256.

**Only then the screens.** The record copies M1–M6 from the bound failed record: the scores, bars and held list.
It labels them as the failed run's one-time screens, applied to the same trained state. Nothing is re-scored. The
recovery record's verdict is ELIGIBLE only on this path.

Why this keeps M1–M6 one-time: the screens were applied once, in entry 73, to the state `adba50fb…`. The replay
adds no new look at either validation set. An exact digest match shows that the saved file is the state that was
screened.

## 4. Outcomes (fixed before any replay)

| Outcome | Reading | What it permits |
|---|---|---|
| Refused at preconditions (gate, pins, code, runtime, memory, cache) | nothing ran | a reviewed decision; no silent repair |
| STOP: selection, start check or fit record differs (**not reproduced**) | the fit is not reproducible here; no challenger | H-MX closes. No second replay, and no re-fit with fresh screens under this card |
| STOP or FAILED at the save or reload | no verified challenger | a reviewed decision; no automatic retry |
| VOID (ceiling, interruption) | no result | a reviewed decision; no automatic retry |
| **COMPLETED, ELIGIBLE** | `adba50fb…` saved, reloaded and verified; M1–M6 as recorded in entry 73 | proposing export and parity (card §8), with its own approval |

Exact reproduction of a full 7,500-update CPU fit across processes has not been measured. The in-process bitwise
repeat is tested on small networks only. The replay is itself that measurement.

## 5. Evidence, gate and ceiling

- **Evidence** (create-only, one run):
  - `docs/superpowers/evidence/htb-mix-recovery/` holds `started.json` and `record.json`, kind
    `htb-mix-recovery-record/1`.
  - `htb-mix-recovery-run/` holds the wrapper's files.
  - The record carries the amendment's sha256, the failed record's sha256 and path, the link to entry 73, the code
    and runtime bindings, the cache preparation, the replayed fit, the comparison, the save and reload, and, on
    success, the screens copied from entry 73.
- **Gate:** `HTB_MIX_RECOVERY_AUTHORIZED` in `scripts/GPU/alphazero/htb_mix_recovery_authorization.py`, created
  closed. It is opened in a one-line launch commit and restored on every exit. The gate inventory becomes 74.
- **Wrapper:** `docs/superpowers/evidence/htb-mix-recovery-run/01_wrapper.sh`. It is derived from the H-MX fit
  wrapper, with only the paths, the gate, the module, the record kind and the subcommand (`run replay`) changed. TERM
  is at 11,690 s and KILL 10 s later (11,700 s), the fit's ceiling. The input check (the student file), the idle
  gate, the background-compute check, the process-group supervisor, `outer.json` and the gate restoration are the
  parent's.
- **Guards:** a clean linked worktree, BOINC quit and Unreal Editor closed. Available memory must be at least 12 GiB.
- **Expected cost:** about 6,400 s. That is the failed run's 6,527 s, less the four validation scorings it skips.

## 6. Export (card §8) now admits only a recovered challenger

`htb_mix_export` admits the challenger only from a recovery record that meets all of the following:

- it is COMPLETED, of this card and amendment, and committed;
- its wrapper evidence is committed and clean;
- `outer.json` shows a clean completion under the 11,700 s ceiling, with the pid and launch commit bound;
- its verdict is ELIGIBLE, with `verified: true` and a reload digest of `adba50fb…`;
- it is bound to the failed record's sha256;
- the challenger file exists and its sha256 equals the recorded one.

The failed fit record alone is not admissible: its status is FAILED. The export's steps and tolerances are unchanged.

## 7. Prediction on record (before any replay; low confidence)

**Exact reproduction: about 75%.** The training path is single-process, fixed-order and CPU-only, and fp32
evaluation-mode BatchNorm makes no random draw. In-process repeats are bitwise on small networks. The risk is
non-deterministic accumulation order in a CPU kernel across processes. This has not been measured at full scale.
If it reproduces, the save and reload verify: about 98%.

## 8. What this amendment does not authorize

- Running the replay, opening any gate, or exporting. Each needs the user's approval.
- A second replay after any outcome, a re-fit with fresh screens, or relaxing the digest match.
- Re-scoring either validation set.
- Editing `htb_mix.py`, `htb_pilot.py`, `search_distillation_d2.py`, any H-OP or H-TB file, or the failed evidence.
- Creating the cache directory by hand.
