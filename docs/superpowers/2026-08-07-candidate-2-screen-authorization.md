# Candidate 2 — 64-Game Mechanics Screen — Operator Authorization

**Date:** 2026-08-07 · **Status:** DRAFT, not yet countersigned · **Scope: one 64-game
mechanics screen. Nothing else.**

Governing design: `docs/superpowers/specs/2026-08-05-competitive-readout-strength-design.md`
(rev 2, §7.4, §8.2, §8.3, §10). Preflight: passed 2026-08-07, recorded in
`docs/alphazero-value-search-experiment-ledger.md`.

This document authorizes exactly one run. Every parameter is frozen; none is a decision
to make at run time.

## What this run is, and is not

A **mechanics screen**. Candidate 2 introduces new root-aware selection code that has
never played a real game, so this run exists to exercise it against a real model and
catch gross harm or operational failure. Unit tests establish mechanics; this catches
what they cannot.

**It cannot promote anything.** Per §8.2 the 64-game screen may only *stop*; no success
decision is taken from it, and it has no early-success path. A pass means only that the
decisive 800-game match becomes eligible — and that match needs its own authorization.

**The preflight established reach and scope, not merit.** Its `6.08%` override rate says
the rule fires often enough to be worth testing and not so often as to be unbounded. It
is **not** evidence that the overridden moves are better, and it must not be reported or
reasoned about as if it were.

**Candidate 1's `+293` Elo is not a prior for this run.** Candidate 1 contrasted argmax
against a control that sampled at `T=1.0` in the opening and `T=0.1` afterwards. Here
**both agents sample the opening identically** and differ only after ply 19. The two
experiments do not share an effect size, and no numeric expectation is carried across.

## Isolation — what differs between the agents

| | plies 0–19 | plies ≥ 20 |
|---|---|---|
| **control** (`opening_then_argmax`) | temperature `1.0` | visit argmax |
| **candidate** (`hoeffding_lcb`) | temperature `1.0` | visit argmax **+ frozen Hoeffding override** |

Both agents sample the opening identically, solely to supply match diversity. **The only
difference is the post-opening override.** The candidates are never stacked: this run
contains no all-ply-argmax contrast, and Candidate 1's readout appears nowhere in it.

## The frozen rule — applied, not revisited

Frozen 2026-08-06 before any telemetry existed:

```
ε(n) = R · sqrt( ln(2/δ) / (2n) )        R = 2, δ = 0.05
LCB_i = q_root_perspective_i − ε(n_i)
play challenger  iff  n_L ≥ 8  and  n_C ≥ 8  and  LCB_C > LCB_L
otherwise play the visit leader
```

No constant, eligibility rule, or threshold may change. Amending any of them voids this
authorization and requires re-freezing before a run.

## The checkpoint

```
checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors
sha1  209cf2d4fd24a48553d259dd71b4954867b9473e
```

Both agents load this one checkpoint; they differ only in readout.

## Frozen parameters

| | |
|---|---|
| candidate readout | `hoeffding_lcb` |
| control readout | `opening_then_argmax` |
| games | **64** (32 per colour, balanced by construction) |
| base seed | **`202608124`** |
| seed interval | **`[202608124, 202608188)`** — half-open, 64 wide |
| prior seed intervals | **`[202608060, 202608124)`** — Candidate 1, consumed |
| simulations | 400 new per move, asserted per ply |
| search | **cold** — fresh root every ply |
| Dirichlet noise | off |
| board / max moves | 24 / 280 |
| workers | 1 |
| replay capture | **required** |

**The new interval is NOT yet reserved.** It becomes `RESERVED` in
`docs/superpowers/2026-08-06-competitive-readout-seed-ledger.md` only when this document
is countersigned, and `CONSUMED` when execution begins. Adding the row before signature
would reserve an interval for a run that may never be authorized.

`[202608124, 202608188)` is adjacent to Candidate 1's `[202608060, 202608124)`, not
overlapping — half-open intervals share no seed at the boundary.
`validate_seed_intervals` will confirm this before any game is played.

**Expected cost:** Candidate 1 measured ~59 s/game single-worker, so 64 games projects to
roughly **63 minutes**. Both agents here are deterministic after ply 19, which may change
game length; treat the estimate as approximate.

## Preconditions

**Code-enforced** — the tooling refuses to proceed:

- **Clean worktree.** `_git_provenance` refuses a dirty tree, with no override and no
  repository parameter.
- **Readable checkpoint.** `_sha1` raises rather than recording `None`.
- **Disjoint seeds.** `validate_seed_intervals` runs before any game and validates the
  whole set pairwise.

**Manual:**

1. **Shipped search unchanged.**

   ```bash
   git diff --quiet d5326a0 -- scripts/GPU/alphazero/mcts.py scripts/GPU/alphazero/self_play.py
   echo "EXIT=$?"     # must be 0
   ```

2. **Suite passes**, measured immediately before the run — not quoted from a document.

3. **No output artifact already exists.**

   ```bash
   for p in logs/eval/candidate2_screen.json \
            logs/eval/candidate2_screen_games.jsonl \
            logs/eval/candidate2_screen.stdout \
            logs/eval/candidate2_screen.exit \
            logs/eval/candidate2_screen_replays; do
     if [ -e "$p" ]; then echo "REFUSE: $p already exists"; exit 1; fi
   done
   echo "output paths clear"
   ```

4. **Checkpoint path and hash confirmed.**

## Procedure

One command, every frozen parameter pinned. Do not vary, add or omit a flag.

```bash
nohup bash -c '.venv/bin/python -m scripts.GPU.alphazero.eval_readout_match \
  --checkpoint checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors \
  --candidate-readout hoeffding_lcb \
  --control-readout opening_then_argmax \
  --games 64 \
  --base-seed 202608124 \
  --prior-seed-interval 202608060:202608124 \
  --board-size 24 \
  --mcts-sims 400 \
  --mcts-eval-batch-size 14 \
  --mcts-stall-flush-sims 48 \
  --opening-temp-plies 20 \
  --temp-high 1.0 \
  --temp-low 0.1 \
  --max-moves 280 \
  --workers 1 \
  --replay-dir logs/eval/candidate2_screen_replays \
  --output logs/eval/candidate2_screen.json
rc=$?
printf "%s\n" "$rc" > logs/eval/candidate2_screen.exit
exit "$rc"' \
  > logs/eval/candidate2_screen.stdout 2>&1 &
disown
```

`--temp-low 0.1` is pinned for completeness but is **unused** by both readouts here:
`opening_then_argmax` forces post-opening temperature to `0.0`, and `hoeffding_lcb`
consults only `temp_high` before ply 20. Pinning it prevents a default drift from
mattering if either mapping ever changes.

`logs/eval/candidate2_screen.exit` is the authoritative status. Do not infer success from
an output file, a notification summary, or a pipeline's exit code.

## A. Immediate runtime aborts

Raise **during** the run; no scorable result is produced. Enforced by `eval_integrity`.

- Illegal move or crash.
- Simulation-budget mismatch at any ply.
- Corrupt required telemetry — non-finite `root_value`, or a `None`/non-finite mean on a
  **visited** child.
- Agent, configuration or colour mis-binding; `unknown_error`; incomplete or duplicate
  results.

An abort is a tooling or configuration fault, **not** a result about the candidate.
Re-running afterwards requires fresh authorization: the seed interval will already be
partly consumed.

## B. Post-run disposition

Evaluated after all 64 games finish, from the completed artifact. **The screen may only
stop.** There is no pass threshold, and no success decision is taken here.

- **Futility (§8.2):** a 95% score-rate interval entirely **below** 50% — at n=64, an
  observed score at or below about **`0.378`** (≈ −87 Elo).

**For Candidate 2, futility CLOSES the readout line** — unlike Candidate 1, where a
futility trigger meant halt-and-investigate. Candidate 2 is the line's only remaining
strength hypothesis; if the frozen rule loses convincingly on real play, the readout line
is finished and the next work is a separately scoped training project.

**Not required at 64 games:** 55% overall, or 45% per colour. That gate would discard
roughly two of every three candidates capable of clearing the 800-game bar while still
advancing about one null in five, which is why §8.2 makes this screen futility-only. Do
not reinstate it after seeing the score.

**Anything short of futility means the 800-game decisive match becomes eligible.** It
does not mean the candidate works, and no Elo estimate from 64 games should be treated as
an effect size — the decisive match exists because 64 games cannot resolve one.

## What to report

From the match artifact and the integrity outcome only.

- Contents of `candidate2_screen.exit`.
- Score rate with its 95% interval, and per-colour intervals.
- Decisive-only rate, termination distribution, state-cap rate — secondary.
- Wall-clock.
- Provenance: commit, worktree-clean flag, checkpoint hash, complete readout configs,
  seed interval and convention, prior intervals, RNG derivation.
- Whether any §A abort fired.

**Telemetry analysis is NOT part of this authorization.** Replays are captured here and
analyzed later, under a separate signature, if at all.

## What remains UNAUTHORIZED

- The **800-game decisive match**, whatever this screen returns.
- Any analysis of this run's captured telemetry.
- Re-running, topping up, extending, or replacing this 64-game block.
- Any change to the frozen rule, its constants, the readouts, or the seed interval.
- Any change to shipped search, self-play, the network, or the checkpoint-tournament
  default.
- Any product change or product strength claim.

## Standing rules

- Amendments and freezes precede the work they govern.
- Baseline counts come from a measured collect, never a document.
- Undefined statistics are `None`/null — never `0`, never `false`.
- Read exit codes from the process, not from a pipe.

---

## Countersignature

Execution is authorized only when this block is filled in, committed, and pushed.
**Unsigned, this document authorizes nothing.**

```
authorizer          : ____________________
timestamp (UTC)     : ____________________
authorization basis : ____________________   # the reviewed commit this signature approves
execution commit    : the commit containing this completed countersignature block
approved scope      : the exact command in "Procedure" above, unmodified —
                      every flag as written, none added, none omitted
```

**Conditions:**

- The run must execute from the **execution commit** — the commit containing this
  completed block — with a clean worktree. Verify after signing:

  ```bash
  git log -1 --format=%H -- docs/superpowers/2026-08-07-candidate-2-screen-authorization.md
  git rev-parse HEAD          # must be the same commit, and the tree must be clean
  ```

- **On signature, add `[202608124, 202608188)` to the seed ledger as `RESERVED`**, before
  launch, so no other authorization can claim an overlapping range. It becomes `CONSUMED`
  when execution begins, and remains `CONSUMED` even if the run aborts.
- Approval covers **one execution**. It does not extend to a re-run, a top-up, a
  parameter change, or a retry after an §A abort.
- Changing any frozen parameter voids this signature. Amend and re-sign **before**
  running, never after seeing a result.
