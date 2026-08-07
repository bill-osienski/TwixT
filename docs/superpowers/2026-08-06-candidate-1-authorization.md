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

Also confirm by hand before starting:

- `git diff --quiet d5326a0 -- scripts/GPU/alphazero/mcts.py scripts/GPU/alphazero/self_play.py`
  exits 0. Shipped search must be unchanged.
- The full suite passes. Last measured at `6d59c1d`: 2,795 passed / 4 skipped /
  53 deselected / 0 failed.

## Procedure

One command. Do not vary it.

```bash
.venv/bin/python -m scripts.GPU.alphazero.eval_readout_match \
  --checkpoint checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors \
  --candidate-readout argmax \
  --control-readout tournament \
  --games 64 \
  --base-seed 202608060 \
  --workers 1 \
  --replay-dir logs/eval/candidate1_diagnostic_replays \
  --output logs/eval/candidate1_diagnostic.json
```

Long runs: `nohup … & disown`. The harness kills foreground jobs, and `setsid` is absent
on macOS. Read the exit status from the process, never from a pipe.

## Stop conditions

Any of these stops the run. The first four are enforced by `eval_integrity`; the run
raises rather than producing a result.

- Illegal move or crash (aborts through the engine / `_WorkerFailed`).
- Simulation-budget mismatch at any ply.
- Corrupt required telemetry — non-finite `root_value`, or a `None`/non-finite mean on a
  **visited** child.
- Agent, configuration or colour mis-binding; `unknown_error`; incomplete or duplicate
  results.
- **Futility (§8.2):** a 95% score-rate interval entirely **below** 50% — at n=64, an
  observed score at or below about **`0.378`** (≈ −87 Elo).

**"Stop" means HALT AND INVESTIGATE, not close the line.** Candidate 1 is a diagnostic
with no 800-game path; a clear argmax loss is a *finding* about visit-leader reliability
and must be understood before Candidate 2's budget is spent (§7.3).

**Not required at 64 games:** 55% overall, or 45% per colour. That gate would discard
roughly two of every three candidates capable of clearing the 800-game bar while still
advancing about one null in five, which is why §8.2 makes this screen futility-only.

## What to report

- Score rate with its 95% interval, and per-colour intervals.
- Decisive-only rate, termination distribution, state-cap rate — secondary.
- Wall-clock, to cost the eventual 800-game run.
- Non-leader selection rates before and after ply 20, so a near-null can be attributed.
  All-ply argmax changes **both** halves; these numbers say which moved. **Descriptive —
  they gate nothing.**
- Provenance block: commit, worktree-clean flag, checkpoint hash, complete readout
  configs, seed interval and convention, prior intervals, RNG derivation.

## What remains UNAUTHORIZED

- **Candidate 2** — its rule and gates are frozen, but no run of it is authorized.
- Any **800-game** match, for either candidate.
- Running the **preflight against real telemetry**. Capture it here; analyzing it needs
  its own authorization.
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

**Countersignature required before execution.** Unsigned, this document authorizes
nothing.
