# Candidate 1 — Operator Authorization

**Date:** 2026-08-06 · **Status:** DRAFT, not yet countersigned · **Scope: one 64-game
all-ply-argmax diagnostic. Nothing else.**

Governing design: `docs/superpowers/specs/2026-08-05-competitive-readout-strength-design.md`
(rev 2, §7.3, §8.2, §8.3, §10). Tooling: `docs/superpowers/plans/2026-08-06-competitive-readout.md`,
11/11 complete at `6d59c1d`.

This document authorizes exactly one run. Every parameter below is frozen; none is a
decision to make at run time.

## What this run is, and is not

Candidate 1 is a **DIAGNOSTIC**. It is not a promotion match and it has **no 800-game
follow-up** under any outcome (§7.3). It cannot promote anything, and its result does
not change the checkpoint-tournament default (policy 2).

Its five purposes:

1. Validate candidate/control identity and colour binding end to end.
2. Validate equal budgets, RNG-stream separation, provenance and replay capture.
3. Sight the effect: grossly positive, near-null, or unexpectedly negative.
4. Measure wall-clock, so the eventual 800-game commitment is costed before it is made.
5. Produce fresh root-child visit/Q telemetry for Candidate 2's **already-frozen**
   preflight.

## Phase A is EXCLUDED from the strength rationale

The `MODEL_PATH` provenance audit
(`docs/superpowers/2026-08-06-model-path-provenance-audit.md`, verdict `MISMATCH`)
established that the product server's `server/model.onnx` **cannot** have come from
`calib020_0001` — the artifact predates that checkpoint by five weeks.

Therefore:

- **No part of this run's rationale rests on the product server**, its readout policy, or
  the three defects Workstream 1 repaired.
- Phase A is a defect-repair workstream carrying **no strength claim**, and nothing
  measured here may be reported as evidence about product play.
- The result speaks only to the Python evaluation harness at the stated configuration.

## The checkpoint — the one input the tooling cannot validate

```
checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors
sha1  209cf2d4fd24a48553d259dd71b4954867b9473e
```

The runner hashes it and records the hash, but it cannot know whether the *right* file
was named. Confirm the path and hash before starting. Both agents load this one
checkpoint; they differ only in readout.

## Frozen parameters

| | |
|---|---|
| candidate readout | `argmax` — all-ply visit argmax, deterministic canonical tie-break |
| control readout | `tournament` — temperature `1.0` through ply 19, then `0.1` |
| games | **64** (32 per colour, balanced by construction) |
| base seed | **`202608060`** |
| seed interval | **`[202608060, 202608124)`** — half-open, 64 wide |
| prior seed intervals | **`[]`** — this is the first entry in the line's seed ledger |
| simulations | 400 new per move, asserted per ply |
| search | **cold** — fresh root every ply |
| Dirichlet noise | off |
| board / max moves | 24 / 280 |
| workers | 1 |
| replay capture | **required** — feeds Candidate 2's preflight |

**Seed-ledger note.** `[202608060, 202608124)` is the first line-specific interval.
Record it. Every later run must pass it via `--prior-seed-interval 202608060:202608124`,
and `validate_seed_intervals` will refuse any overlap before a game is played.

## Preconditions, enforced in code

These are not reminders; the tooling refuses to proceed without them.

- **Clean worktree.** `_git_provenance` refuses a dirty tree outright, with no override
  and no repository parameter. Commit or stash first.
- **Readable checkpoint.** `_sha1` raises rather than recording `None`.
- **Disjoint seeds.** `validate_seed_intervals` runs before any game.
- All three run **before the first game**, so a misconfiguration costs zero GPU time.

### Manual preconditions — NOT code-enforced

These four are the operator's responsibility. Unlike the three above, nothing in the
tooling refuses the run if they are skipped.

**1. Shipped search unchanged.**

```bash
git diff --quiet d5326a0 -- scripts/GPU/alphazero/mcts.py scripts/GPU/alphazero/self_play.py
echo "EXIT=$?"     # must be 0
```

**2. Suite passes.** Last measured at `6d59c1d`: 2,795 passed / 4 skipped / 53 deselected
/ 0 failed.

**3. No output artifact already exists.** This block may be run once and may not be
topped up or re-run, so a pre-existing artifact means either a stale file that would be
silently overwritten, or an unauthorized earlier attempt. Either way, stop and
investigate — do not delete and proceed.

```bash
for p in logs/eval/candidate1_diagnostic.json \
         logs/eval/candidate1_diagnostic_games.jsonl \
         logs/eval/candidate1_diagnostic.stdout \
         logs/eval/candidate1_diagnostic.exit \
         logs/eval/candidate1_diagnostic_replays; do
  if [ -e "$p" ]; then echo "REFUSE: $p already exists"; exit 1; fi
done
echo "output paths clear"
```

**4. Checkpoint path and hash confirmed** against the block above.

## Procedure

One command, with **every frozen parameter pinned explicitly**. Defaults are not relied
on anywhere: a default that drifts must not silently change an authorized experiment.
Do not vary, add or omit a flag.

```bash
nohup bash -c '.venv/bin/python -m scripts.GPU.alphazero.eval_readout_match \
  --checkpoint checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors \
  --candidate-readout argmax \
  --control-readout tournament \
  --games 64 \
  --base-seed 202608060 \
  --board-size 24 \
  --mcts-sims 400 \
  --mcts-eval-batch-size 14 \
  --mcts-stall-flush-sims 48 \
  --opening-temp-plies 20 \
  --temp-high 1.0 \
  --temp-low 0.1 \
  --max-moves 280 \
  --workers 1 \
  --replay-dir logs/eval/candidate1_diagnostic_replays \
  --output logs/eval/candidate1_diagnostic.json
rc=$?
printf "%s\n" "$rc" > logs/eval/candidate1_diagnostic.exit
exit "$rc"' \
  > logs/eval/candidate1_diagnostic.stdout 2>&1 &
disown
```

`rc` is captured immediately, written, and then re-raised with `exit "$rc"`. Without
that final `exit`, the wrapper would return the status of the `printf` — normally zero —
so a failed run would look successful to anything watching the background process.
Both the file and the process status are now truthful. (`rc`, not `status`: `status` is
read-only in zsh, and the assignment aborts the shell.)

**`--prior-seed-interval` is deliberately absent**, and that absence is itself a frozen
parameter: it encodes the empty prior set `[]`, this being the first entry in the line's
seed ledger. It is the only CLI flag not pinned above. Every subsequent run in this line
must pass `--prior-seed-interval 202608060:202608124` plus any later intervals.

The harness kills foreground jobs and `setsid` is absent on macOS, hence `nohup` +
`disown`. The `bash -c` wrapper exists so the **process's own exit code** is captured:

```bash
cat logs/eval/candidate1_diagnostic.exit     # 0 = success; anything else = failed run
```

That file is the authoritative status. Do not infer success from the presence of an
output file, from a notification summary, or from a pipeline's exit code.

## A. Immediate runtime aborts

These raise **during** the run. No result is produced; the artifact is absent or partial
and must not be scored. All are enforced by `eval_integrity`.

- Illegal move or crash (aborts through the engine / `_WorkerFailed`).
- Simulation-budget mismatch at any ply.
- Corrupt required telemetry — non-finite `root_value`, or a `None`/non-finite mean on a
  **visited** child.
- Agent, configuration or colour mis-binding; `unknown_error`; incomplete or duplicate
  results.

An abort is a **tooling or configuration fault**, not a result about the candidate. Fix
the cause; re-running this block afterwards requires a fresh authorization, because the
seed interval will already have been partially consumed.

## B. Post-run disposition

Evaluated **after all 64 games finish**, from the completed artifact. This is a reading
of the result, not something that halts the run midway.

- **Futility (§8.2):** a 95% score-rate interval entirely **below** 50% — at n=64, an
  observed score at or below about **`0.378`** (≈ −87 Elo).

**Futility here means HALT AND INVESTIGATE, not close the line.** Candidate 1 is a
diagnostic with no 800-game path; a clear argmax loss is a *finding* about visit-leader
reliability and must be understood before Candidate 2's budget is spent (§7.3).

**Not required at 64 games:** 55% overall, or 45% per colour. That gate would discard
roughly two of every three candidates capable of clearing the 800-game bar while still
advancing about one null in five, which is why §8.2 makes this screen futility-only. Do
not reinstate it after seeing the score.

## What to report

**From the match artifact and the integrity outcome only.** Everything below is already
in `candidate1_diagnostic.json` or the exit file; no analysis tool is run.

- Contents of `candidate1_diagnostic.exit`.
- Score rate with its 95% interval, and per-colour intervals.
- Decisive-only rate, termination distribution, state-cap rate — secondary.
- Wall-clock, to cost the eventual 800-game run.
- Provenance block: commit, worktree-clean flag, checkpoint hash, complete readout
  configs, seed interval and convention, prior intervals, RNG derivation.
- Whether any §A abort fired.

**Telemetry analysis is NOT part of this authorization.** The replays are *captured*
here and analyzed later. That includes the non-leader selection rates §7.3 calls for to
attribute a near-null, and the entire preflight. Running `readout_preflight` against
these replays — or computing its statistics by hand — needs its own authorization.
Capture now, analyze under a separate signature.

## What remains UNAUTHORIZED

- **Candidate 2** — its rule and gates are frozen, but no run of it is authorized.
- Any **800-game** match, for either candidate.
- **Any analysis of the captured telemetry**, including `readout_preflight`, the
  non-leader selection rates, and any hand computation of the same statistics. Capture
  here; analyze under a separate signature.
- Re-running, topping up, extending, or replacing this 64-game block.
- Any change to shipped search, self-play, the network, or the checkpoint-tournament
  default.
- Any product change, `model.onnx` re-export, or product strength claim.
- Choosing a different seed interval, or reusing this one.

## Standing rules

- Amendments and freezes precede the work they govern. If this run needs a parameter
  changed, amend this document **before** running, not after seeing a result.
- Baseline counts come from a measured collect, never a number quoted in a document.
- Undefined statistics are `None`/null — never `0`, never `false`.
- Read exit codes from the process, not from a pipe.

---

## Countersignature

Execution is authorized only when this block is filled in, committed, and pushed.
**Unsigned, this document authorizes nothing.**

```
authorizer          : ____________________
timestamp (UTC)     : ____________________
authorization basis : 1654426              # the reviewed state this signature approves
execution commit    : the commit containing this completed countersignature block
approved scope      : the exact command in "Procedure" above, unmodified —
                      every flag as written, none added, none omitted
```

**Why `execution commit` is named this way and not written as a hash.** Committing the
completed block changes the hash, so a document cannot contain the hash of the commit
that contains it. `authorization basis` pins the reviewed content; the execution commit
is whatever commit results from recording the signature, and it is identified by
containment rather than by a value written in advance.

**Conditions attached to the signature:**

- The run must execute from the **execution commit** — the commit that contains this
  completed block — with a clean worktree. `_git_provenance` records the commit it
  actually ran from; if that does not match the commit containing this signature, the
  result is not covered by this authorization. Verify after signing with:

  ```bash
  git log -1 --format=%H -- docs/superpowers/2026-08-06-candidate-1-authorization.md
  git rev-parse HEAD          # must be the same commit, and the tree must be clean
  ```
- Approval covers **one execution**. It does not extend to a re-run, a top-up, a
  parameter change, or a retry after an §A abort.
- Changing any frozen parameter voids this signature. Amend and re-sign **before**
  running, never after seeing a result.

Commit and push this signed state **before** execution, so the authorization exists in
history independently of the run's outcome.
