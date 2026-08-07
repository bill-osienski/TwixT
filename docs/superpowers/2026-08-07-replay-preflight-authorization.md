# Candidate 1 Replay Analysis — Operator Authorization

**Date:** 2026-08-07 · **Status:** DRAFT, not yet countersigned
**Scope: exactly one read-only execution of the frozen preflight analyzer over the 64
Candidate 1 replay sidecars. Nothing else.**

Governing design: `docs/superpowers/specs/2026-08-05-competitive-readout-strength-design.md`
(rev 2, §7.3, §7.4). Inputs produced by
`docs/superpowers/2026-08-06-candidate-1-authorization.md`, run at `d2aaf4f`.

**No GPU work. No checkpoint load. No games.** This reads 64 JSON files and computes
frozen statistics over them.

## Why one authorization, not two

Descriptive outputs and the gate verdict come from the *same* frozen analyzer in the
*same* pass. Splitting them would authorize two executions of one tool over one input
set, adding ceremony without adding control — and would create the opportunity to see the
descriptive numbers before deciding whether to compute the gates, which is exactly the
ordering the freeze exists to prevent.

## What is authorized

One execution of `scripts/GPU/alphazero/readout_preflight.py` producing, in a single pass:

1. **Integrity validation of all 64 sidecars** — schema version, telemetry completeness,
   agent presence. Any failure raises and aborts; see §A.
2. **The frozen population** — post-opening turns (`ply ≥ 20`) belonging to Candidate 1's
   **argmax agent only** (`--agent-id candidate`). Every such turn is in the denominator;
   an ineligible turn counts as "no override" and does not disappear.
3. **The frozen Candidate 2 rule** — Hoeffding LCB over the top two root children by
   completed visits, `ε(n) = R·sqrt(ln(2/δ)/(2n))`, `R = 2`, `δ = 0.05`, `n_min = 8`,
   frozen 2026-08-06 before any telemetry existed.
4. **The frozen preflight gates** — §B below.
5. **Descriptive statistics** — non-leader selection rates split at the opening boundary,
   override rate by ply bucket, challenger visits at override, per-game override counts,
   and colour split.

## What is NOT authorized

- **Any change** to the formula, its constants, the population definition, or any
  threshold. They are frozen; this run applies them, it does not revisit them.
- Re-running the analyzer, or running it against any other replay set.
- Candidate 2's 64-game screen, or any match.
- Any checkpoint load, GPU work, or new game generation.
- Reserving or consuming a seed interval. **This run consumes none** — no games are
  played. The seed ledger is unchanged by it.
- Acting on the descriptive statistics as if they were gates. Colour split in particular
  is descriptive only and may never close or open the candidate.

## Preconditions

**Code-enforced** — the analyzer raises rather than proceeding:

- Every sidecar is `schema_version == 2`.
- Every analyzed turn carries `n_legal ≥ 1` and a list-valued `top2` of exactly
  `min(2, n_legal)` entries. A one-entry list is accepted only when `n_legal == 1`.
- **Every** selected replay contains the named agent; partial absence raises with the
  missing `game_idx` values named.

**Manual** — the operator's responsibility:

1. **Inputs are the Candidate 1 artifacts, unmodified.** Record the count and a checksum
   over the sorted sidecar set before running, so the analyzed inputs are provably the
   ones that run produced.

   ```bash
   ls logs/eval/candidate1_diagnostic_replays/game_*.json | wc -l    # must be 64
   find logs/eval/candidate1_diagnostic_replays -name 'game_*.json' | sort | \
     xargs shasum -a 256 | shasum -a 256
   ```

2. **Output paths clear.** This may run once.

   ```bash
   for p in logs/eval/candidate1_preflight.json logs/eval/candidate1_preflight.exit; do
     if [ -e "$p" ]; then echo "REFUSE: $p already exists"; exit 1; fi
   done
   echo "output paths clear"
   ```

3. **Worktree clean**, and HEAD recorded with the result.

## Procedure

One command, every parameter pinned. No defaults relied on.

```bash
.venv/bin/python -m scripts.GPU.alphazero.readout_preflight \
  --replay-glob 'logs/eval/candidate1_diagnostic_replays/game_*.json' \
  --agent-id candidate \
  --opening-temp-plies 20 \
  --output logs/eval/candidate1_preflight.json
rc=$?
printf "%s\n" "$rc" > logs/eval/candidate1_preflight.exit
```

`--agent-id candidate` selects Candidate 1's **argmax** agent: the run bound
`candidate → argmax` and `control → tournament`, so this is the frozen post-opening
argmax-agent population.

This is seconds of work on 64 files, so it runs in the foreground; `rc` is still captured
because the exit code carries the verdict.

| exit | meaning |
|---:|---|
| `0` | all frozen gates passed |
| `2` | a frozen gate closed Candidate 2 |
| anything else | **§A integrity abort** — the analyzer raised; there is no verdict |

## §A. Integrity aborts

The analyzer raises and produces no report. **An abort is not a verdict**, and the
statistics from a partial pass must not be read or reported.

- A sidecar whose `schema_version` is not 2.
- Missing, non-list, or wrong-length `top2`, or missing `n_legal`, at any analyzed turn —
  named with its game and ply.
- Any selected replay lacking the named agent.

An abort means the inputs or the tooling are wrong, not that Candidate 2 failed. Fix the
cause; re-running needs a fresh authorization.

## §B. Frozen verdict

Applied by `evaluate_gates`. **Not revisable by this authorization.**

| statistic | closes Candidate 2 when | reason |
|---|---|---|
| override rate | `< 0.5%` | insufficient reach to justify the spend |
| override rate | `> 15%` | excessive scope — not the conservative occasional rule hypothesized |
| share of overrides in one game | `> 50%` | concentration — the rate describes one abnormal trajectory |
| rows with corrupt Q on a visited child | any | halt; telemetry defect, not a candidate result |
| population | empty | no basis for any verdict |
| **colour split** | **never** | **DESCRIPTIVE ONLY** |

**Equality at a boundary PASSES.** The frozen operators are strict — `< 0.5%`, `> 15%`,
`> 50%` — so a statistic landing exactly on `0.005`, `0.15`, or `0.50` is inside the
band. This is a restatement of the frozen operators, not a new decision.

The first three thresholds are **pre-registered judgement bounds** chosen before any
telemetry existed. They are not derived from data, and the `< 0.5%` floor in particular
is a **spending** judgement — not a claim that a rare override cannot be decisive.

## What to report

- Contents of `candidate1_preflight.exit`, and the verdict it encodes.
- `population_plies`, `eligible_plies`, `overrides`, `override_rate`.
- `max_single_game_share`, `games_with_overrides`, `undefined_q_plies`.
- Which gates, if any, failed.
- Descriptive: non-leader selection split at ply 20, override rate by ply bucket,
  challenger visits at override, per-game override counts, colour split — each labelled
  descriptive.
- HEAD, worktree state, and the sidecar-set checksum from precondition 1.

## Afterward

**If a gate closes Candidate 2 (`exit 2`):** record the result in the experiment ledger
and **close Candidate 2 without spending any GPU time**. That is a successful outcome of
the preflight, not a disappointment — it is the mechanism working as designed.

**If all gates pass (`exit 0`):** record the result, then draft a **separate**
authorization for Candidate 2's 64-game mechanics screen. That authorization must reserve
a **new** seed interval and pass Candidate 1's consumed interval as a prior:

```
--prior-seed-interval 202608060:202608124
```

Passing this run's gates authorizes nothing further by itself.

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
  git log -1 --format=%H -- docs/superpowers/2026-08-07-replay-preflight-authorization.md
  git rev-parse HEAD          # must be the same commit, and the tree must be clean
  ```

- Approval covers **one execution**. It does not extend to a re-run, a different replay
  set, a parameter change, or a retry after an §A abort.
- Changing any frozen parameter voids this signature. Amend and re-sign **before**
  running, never after seeing a result.
