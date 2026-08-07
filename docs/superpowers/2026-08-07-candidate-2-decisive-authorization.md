# Candidate 2 — 800-Game Decisive Match — Operator Authorization

**Date:** 2026-08-07 · **Status:** DRAFT, not yet countersigned · **Scope: one 800-game
decisive match. Nothing else.**

Governing design: `docs/superpowers/specs/2026-08-05-competitive-readout-strength-design.md`
(rev 2, §7.4, §8.1, §8.3, §10). Prerequisites, both recorded in
`docs/alphazero-value-search-experiment-ledger.md`: preflight passed 2026-08-07
(override rate 6.08%); 64-game mechanics screen ran 2026-08-07 and did not trigger
futility.

This document authorizes exactly one run. Every parameter is frozen.

## What this run is

**The only run in this line that can resolve the question.** The 64-game screen could
only stop; this is the first and only stage with a promotion bar. It exists precisely
because 64 games cannot separate harm, null, and useful gain.

## The prior is adverse, and that is recorded here deliberately

The mechanics screen lost **28–36**, score rate `0.4375`, CI95 `[0.316, 0.559]`,
`−43.7` Elo with an interval spanning `−134` to `+41`. Promotion below requires an
observed score rate around `0.535`. **That is a large gap from the screen's point
estimate, and expectations should be modest.**

It is written here rather than left implicit so that a negative outcome is not later
narrated as a surprise. But it is **not a gate**: the screen was designed futility-only,
and futility did not fire. Declining to run on the strength of a 64-game point estimate
would install exactly the post-result gate that design excluded, and would leave
Candidate 2 unresolved after it cleared every frozen prerequisite.

**If the ~13-hour cost is independently unacceptable**, the correct disposition is
**"decisive match declined for resource cost"** — never "Candidate 2 failed." Those are
different records and only one of them is true.

## What this run is NOT

- **Not a ship decision.** §8.1's bar is a **research promotion bar**. No deployment
  target exists for a Q-informed readout: policy 3 is defined as all-ply visit argmax and
  cannot receive this rule until its target is explicitly redefined in writing, and no
  Python result adopts into the product (policy 1).
- **Not evidence about the product.** `server/model.onnx` serves a different network
  (`MISMATCH`, `docs/superpowers/2026-08-06-model-path-provenance-audit.md`).
- **Not a change to the checkpoint-tournament default** (policy 2), whatever it returns.

## Isolation — unchanged from the screen

| | plies 0–19 | plies ≥ 20 |
|---|---|---|
| **control** (`opening_then_argmax`) | temperature `1.0` | visit argmax |
| **candidate** (`hoeffding_lcb`) | temperature `1.0` | visit argmax **+ frozen Hoeffding override** |

Both agents sample the opening identically for match diversity; the only difference is the
post-opening override. Post-opening visit argmax is deliberately both the control's
readout and the candidate's fallback when the rule declines.

## The frozen rule — applied, not revisited

```
ε(n) = R · sqrt( ln(2/δ) / (2n) )        R = 2, δ = 0.05
LCB_i = q_root_perspective_i − ε(n_i)
play challenger  iff  n_L ≥ 8  and  n_C ≥ 8  and  LCB_C > LCB_L
otherwise play the visit leader
```

Frozen 2026-08-06 before any telemetry existed. No constant, eligibility rule, or
threshold may change.

## Frozen parameters

| | |
|---|---|
| checkpoint | `checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors`, sha1 `209cf2d4…` |
| candidate readout | `hoeffding_lcb` |
| control readout | `opening_then_argmax` |
| games | **800** (400 per colour, balanced by construction) |
| base seed | **`202608188`** |
| seed interval | **`[202608188, 202608988)`** — half-open, 800 wide |
| prior intervals | **`[202608060, 202608124)`** and **`[202608124, 202608188)`** |
| simulations | 400 new per move, asserted per ply |
| search | **cold** — fresh root every ply |
| Dirichlet noise | off |
| board / max moves | 24 / 280 |
| workers | 1 |
| replay capture | **required** |

**No top-up, ever.** If the run aborts, the interval is consumed and a continuation needs
a fresh authorization with a *new* interval — not an extension of this one.

**The interval is NOT yet reserved.** On countersignature, add this row verbatim to
`docs/superpowers/2026-08-06-competitive-readout-seed-ledger.md` **in the same commit**:

```
| 3 | `[202608188, 202608988)` | 800 | Candidate 2 — 800-game decisive match | 2026-08-07 | execution commit containing decisive-match countersignature | **RESERVED** |
```

After the run, the result-recording commit replaces the phrase with the real execution
hash and flips `RESERVED` → `CONSUMED`.

**Expected cost:** the screen measured `57.9 s/game` single-worker, so 800 games projects
to roughly **12.9 hours**. Plan for an overnight run.

## Statistics — the bar, written before the run

```
n = 800   SE ≈ 0.0177   95% half-width ≈ 0.0347
promotion requires an observed score rate ≳ 0.535   ≈  +24 Elo
~80% power at a true effect of ≈ 0.550             ≈  +35 Elo
```

**This is the research promotion bar.** A candidate not visible in 800 games is not worth
the reliability risk of adopting, which is why the design makes the minimum detectable
effect the bar rather than treating low power as a reason for a weaker test.

### Promotion requires ALL of

- **Primary:** draw-inclusive score rate with a 95% **lower** bound above 50%.
- **Colour safety:** reject only if either colour's own 95% **upper** bound falls below
  50% — at 400 games per colour, an observed colour score at or below about `0.451`
  (≈ −34 Elo), i.e. convincing one-sided harm. **No colour-gap veto.**
- **Zero-tolerance integrity** (§A) — none fired.
- **Search-identity evidence** confirming only final move selection differed.

**Secondary, reported but never decisive:** decisive-only score rate and state-cap count.
Secondary because excluding draws biases the comparison if the candidate changes draw
propensity.

Do not add, relax, or reinterpret any of these after seeing the score.

## Preconditions

**Code-enforced:** clean worktree (`_git_provenance`, no override, no repository
parameter); readable checkpoint (`_sha1` raises rather than recording `None`); disjoint
seeds (`validate_seed_intervals`, whole set pairwise, before any game).

**Manual:**

1. **Shipped search unchanged.**

   ```bash
   git diff --quiet d5326a0 -- scripts/GPU/alphazero/mcts.py scripts/GPU/alphazero/self_play.py
   echo "EXIT=$?"     # must be 0
   ```

2. **Suite passes**, measured immediately before the run — not quoted from a document.

3. **No output artifact already exists.**

   ```bash
   for p in logs/eval/candidate2_decisive.json \
            logs/eval/candidate2_decisive_games.jsonl \
            logs/eval/candidate2_decisive.stdout \
            logs/eval/candidate2_decisive.exit \
            logs/eval/candidate2_decisive_replays; do
     if [ -e "$p" ]; then echo "REFUSE: $p already exists"; exit 1; fi
   done
   echo "output paths clear"
   ```

4. **Checkpoint path and hash confirmed.**

5. **Search identity — the promotion condition, established explicitly.** §8.1 requires
   evidence that only final move selection differed. The full suite contains these tests,
   but this artifact must record them in its own right, so run them targeted and keep the
   exit status:

   ```bash
   .venv/bin/python -m pytest -q \
     tests/test_eval_readout_telemetry.py::test_search_identity_across_two_independent_searches \
     tests/test_eval_readout_telemetry.py::test_search_identity_test_can_actually_fail \
     tests/test_eval_readout_telemetry.py::test_readout_cannot_advance_the_search_rng
   echo "EXIT=$?"     # must be 0
   ```

   The three carry distinct weight: the first shows two independent searches at the same
   seed produce identical counts, root value and top-two telemetry; the second shows a
   *different* seed does change the tree, so the first is not vacuous; the third shows the
   eval readout leaves `mcts.rng` untouched while `mcts.select_move` demonstrably advances
   it. Together they are what makes "only the readout differed" a measured claim rather
   than an assertion.

6. **Disk headroom for ~800 replay sidecars.** Measured on the screen's output: 64
   sidecars occupy **1.9 MB**, about 29.7 KB each, so 800 project to roughly **23 MB**.
   Free space at drafting was ~112 GB, so headroom is ample — confirm before a 13-hour
   run rather than assuming it.

## Procedure

One command, every frozen parameter pinned. Do not vary, add or omit a flag.

```bash
nohup bash -c '.venv/bin/python -m scripts.GPU.alphazero.eval_readout_match \
  --checkpoint checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors \
  --candidate-readout hoeffding_lcb \
  --control-readout opening_then_argmax \
  --games 800 \
  --base-seed 202608188 \
  --prior-seed-interval 202608060:202608124 \
  --prior-seed-interval 202608124:202608188 \
  --board-size 24 \
  --mcts-sims 400 \
  --mcts-eval-batch-size 14 \
  --mcts-stall-flush-sims 48 \
  --opening-temp-plies 20 \
  --temp-high 1.0 \
  --temp-low 0.1 \
  --max-moves 280 \
  --workers 1 \
  --replay-dir logs/eval/candidate2_decisive_replays \
  --output logs/eval/candidate2_decisive.json
rc=$?
printf "%s\n" "$rc" > logs/eval/candidate2_decisive.exit
exit "$rc"' \
  > logs/eval/candidate2_decisive.stdout 2>&1 &
disown
```

`logs/eval/candidate2_decisive.exit` is the authoritative status. Do not infer success
from an output file, a notification summary, or a pipeline's exit code.

**Do not inspect partial results while the run is in progress.** A peek at the running
score is an unplanned interim look, and acting on one would break the single-decision
design this bar assumes.

## §A. Immediate runtime aborts

Raise during the run; no scorable result is produced. Enforced by `eval_integrity`:
illegal move or crash; simulation-budget mismatch; corrupt required telemetry;
agent/configuration/colour mis-binding; `unknown_error`; incomplete or duplicate results.

An abort is a tooling or configuration fault, **not** a result about the candidate. The
interval is consumed regardless; a re-run needs a fresh authorization and a new interval.

## What to report

From the match artifact and the integrity outcome only.

- Contents of `candidate2_decisive.exit`.
- Score rate with its 95% interval, and per-colour intervals with their **upper** bounds
  against the colour-safety rule.
- Decisive-only rate, termination distribution, state-cap rate — secondary.
- Wall-clock.
- Provenance: commit, worktree-clean flag, checkpoint hash, complete readout configs,
  seed interval and convention, prior intervals, RNG derivation.
- Whether any §A abort fired.
- **The search-identity evidence**: the exit status of the three targeted tests from
  precondition 5, reported alongside the protected-file diff. §8.1 lists this as a
  promotion requirement, so the artifact must show it was satisfied rather than assumed.
- The promotion verdict against §8.1, stated as met or not met — never as a near-miss
  that might justify a follow-up.

**Telemetry analysis is NOT part of this authorization.**

## What remains UNAUTHORIZED

- Any top-up, extension, re-run or replacement of this 800-game block.
- Any analysis of its captured telemetry.
- Any change to the frozen rule, readouts, thresholds, or seed interval.
- Adoption into policy 1 (product), policy 2 (tournament default), or policy 3 (which
  cannot receive this rule until its target is redefined in writing).
- Any change to shipped search, self-play, the network, or the product.

## Afterward

- **Promotion bar met:** record it, and note that promotion is *research* promotion. Any
  adoption requires policy 3's target to be redefined in writing first, and separately a
  JavaScript/product transfer test before any product claim.
- **Bar not met:** record it and **close the readout line**. Candidate 2 is its last
  strength hypothesis; the next work is a separately scoped training project. Do not
  propose a third readout formula, a relaxed bar, or a larger match to rescue it.
- **§A abort:** record a tooling fault, not a candidate result.

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

- The run must execute from the **execution commit** with a clean worktree. Verify after
  signing:

  ```bash
  git log -1 --format=%H -- docs/superpowers/2026-08-07-candidate-2-decisive-authorization.md
  git rev-parse HEAD          # must be the same commit, and the tree must be clean
  ```

- **Add the reservation row above to the seed ledger in the same commit as this
  signature.**
- Approval covers **one execution**. It does not extend to a re-run, a top-up, a
  parameter change, or a retry after an §A abort.
- Changing any frozen parameter voids this signature. Amend and re-sign **before**
  running, never after seeing a result.
