# H-MX export amendment 1: export attempt 2, with fresh evidence and a Node dependency preflight

**Status:** DRAFT for review (2026-11-20). The amendment and its model-free build are approved; the launch is not.
The build pins this file by sha256. Any change after review is re-pinned and rebuilt before any gate opens.

**Amends:** the H-MX card, `docs/superpowers/2026-11-18-htb-mix-distillation-card.md`, revision 1 (`d2c9b787…`),
which stays frozen and unedited, as recovery amendment 1 (`8d6e3866…`) left it. This amendment changes only where the
export stage (card §8, step 4 of §14) writes its evidence, the launch's dependency checks, and which export record the
match stages read. The exporter, its invocation, the graph check, both runtime canaries, S1–S4, V3, every tolerance,
the 3,600 s ceiling, the gate and the challenger admission are unchanged.

**Links:** ledger Post-closure entry 75 and do-not-repeat #215 (attempt 1, VOID); entry 74 and #214 (the verified
challenger).

## 1. Attempt 1, preserved

Attempt 1 crashed before any export step (entry 75). The launch worktree had no `node_modules`, so
`htb_onpolicy_export.runtime_versions` raised `FileNotFoundError` reading
`node_modules/onnxruntime-node/package.json`. No exporter, canary or parity step ran, and nothing was written under
`~/twixt-cache/htb-mix` except what the recovery had already saved. Its evidence stays as written:

| File | sha256 |
|---|---|
| `docs/superpowers/evidence/htb-mix-export/record.json` (wrapper-written, `status: VOID`) | `127f0607…4febb` |
| `docs/superpowers/evidence/htb-mix-export-run/outer.json` (inner and outer exit 1, CRASHED) | `7dab1072…bcf7` |

Attempt 2 binds both files by these hashes and their commit, and refuses to start if either is missing,
uncommitted or changed. Nothing in attempt 2 writes under `htb-mix-export/` or `htb-mix-export-run/`. Attempt 1's
own entry (`htb_mix_export run export`) and wrapper stay as built, and they refuse because their evidence exists.

## 2. Attempt 2: fresh, create-only evidence

- **Entry:** `htb_mix_export run export-2`. It runs the same `run_export` body: the same recovery admission
  (amendment 1 §6), exporter command, graph signature check, content-addressed placement under
  `~/twixt-cache/htb-mix/export/<id>/`, manifest, qualification commands and verdict rule. Only the evidence
  directory changes, and the record gains the fields below.
- **Evidence (create-only, one run):**
  - `docs/superpowers/evidence/htb-mix-export-2/`: the record (`htb-mix-export-record/1`, as before), the step logs and
    the gzipped parity sides.
  - `docs/superpowers/evidence/htb-mix-export-2-run/`: the wrapper's files.
- **The record adds:** `attempt: 2`; this amendment's sha256 (`export_amendment_sha256`); the attempt-1 binding
  (both paths and hashes, and its status VOID); and the onnxruntime-node version and the path it resolved from.
- **Gate:** `HTB_MIX_EXPORT_AUTHORIZED`, unchanged. It is opened in a one-line launch commit and restored on every
  exit. No new gate is added; the inventory stays 74.
- **Ceiling:** TERM 3,590 s, KILL 10 s later (3,600 s), the card's.
- **Order before anything is exported:**
  1. the gate, card and TF32 preconditions;
  2. this amendment's pin and commit;
  3. the attempt-1 binding;
  4. the Node dependency (§3);
  5. the empty evidence, export and scratch paths;
  6. the recovery admission.

  A failure at any of these refuses before the record opens. No exporter or Node step runs, and no scratch or export
  directory is created.

## 3. The Node dependency

A linked worktree has no `node_modules`, because git ignores it. H-OP's export, smoke and match launches cloned it
(entries 70–72); attempt 1 did not.

**Launch procedure, before the gate-opening commit,** in the new worktree `<W>`:

1. Clone, don't link: `cp -c -R /Users/bill/projects/TwixT_Game/node_modules <W>/node_modules`. This is an APFS
   clone; a symlink is refused.
2. From `<W>`, resolve the dependency the way the server does, from `server/inference.js`:

   ```sh
   node -e 'const r = require("module").createRequire(process.argv[1]); const p = r.resolve("onnxruntime-node/package.json"); console.log(p); console.log(require(p).version)' "$PWD/server/inference.js"
   ```

   It must print `<W>/node_modules/onnxruntime-node/package.json` and then `1.23.2`. A path outside `<W>` means
   Node found another tree's `node_modules`, and the launch stops.
3. `git status --porcelain` is empty (`node_modules` is ignored).

Only then is the gate-opening commit made and the wrapper started.

**The wrapper preflight** (`docs/superpowers/evidence/htb-mix-export-2-run/01_wrapper.sh`) repeats the check. It sits
among the wrapper's preconditions, after the one-shot checks and before the idle gate:

- `node` is on the PATH;
- `node_modules` is a real directory in the tree, not a symlink;
- Node, from `$W/server/inference.js`, resolves `onnxruntime-node/package.json` to exactly
  `$W/node_modules/onnxruntime-node/package.json`;
- its version is `1.23.2`.

Any miss **refuses cleanly**. The wrapper prints the reason and exits 1. It writes no evidence file and no
`outer.json`, launches nothing, and restores the gate. The idle gate and the BOINC check are not reached.

**The entry repeats it in Python** before the record opens. `node_modules/onnxruntime-node/package.json` must exist
under the root and declare `1.23.2`; otherwise it refuses (exit 2). That covers a launch without the wrapper.
`runtime_versions` is then read as before.

The wrapper is derived from attempt 1's wrapper. The changes are the two evidence paths, the subcommand
(`run export-2`) and the inserted preflight. The idle gate, the BOINC check, the input checked, the supervisor,
`outer.json` and the gate restoration are attempt 1's, unchanged.

## 4. The match stages read attempt 2 only

Both match wrappers, the driver (`tests/htb_integration/mix_match.mjs`) and the analyser
(`tests/htb_integration/mix_match_analyse.mjs`) read the export record at
`docs/superpowers/evidence/htb-mix-export-2/record.json` and its wrapper evidence at
`docs/superpowers/evidence/htb-mix-export-2-run/`.

Besides the existing checks (COMPLETED, PARITY_PASS, the challenger's files and manifest, committed and clean
evidence, a clean `outer.json` under the 3,600 s ceiling, pid and launch commit bound), they now require `attempt: 2`
and this amendment's sha256 in the record.

**Attempt 1 cannot admit a match.** It is at the old path, which nothing reads now. Its record is VOID, has no verdict
and carries neither field. The analyser's sha256 changes, and both match wrappers re-pin it at derivation.

## 5. Outcomes (fixed before the launch)

| Outcome | Reading | What it permits |
|---|---|---|
| Refused (preconditions, attempt-1 binding, Node dependency, admission) | nothing ran | a reviewed decision; no silent repair |
| STOP at the export, the graph check, a canary, S1–S4 or V3 | the challenger is not qualified | nothing; no retry, re-export or tolerance change; a reviewed decision |
| VOID (ceiling, crash, interruption) | no result | nothing; no automatic retry |
| **COMPLETED, PARITY_PASS** | the export is the verified challenger and qualifies | proposing the match smoke (card §9, §11), with its own approval |

There is no third attempt under this amendment.

## 6. What this amendment does not authorize

- Launching attempt 2, opening any gate, or any match stage. Each needs the user's approval.
- Editing the exporter or the qualification tools and tolerances, `htb_mix.py`, `htb_onpolicy_export.py`, any H-OP
  or H-TB file, the recovery evidence, the challenger, or attempt 1's evidence.
- A symlinked `node_modules`, or a different onnxruntime-node version.
- Changing `HARD_MODEL_ID` or the product manifest.
