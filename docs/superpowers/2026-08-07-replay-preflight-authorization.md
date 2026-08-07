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

1. **The analyzed bytes match the countersigned input set.** The digest below is
   **frozen in this document before signature**. Recompute it at execution and compare;
   a mismatch means the inputs are not what was authorized, and the run must not proceed.

   ```
   file count : 64
   set digest : a4e2bfc66fbaa18ff752a6bacb2f449608fa75105ab3a26224ac0cc73516dc7d
   ```

   ```bash
   ls logs/eval/candidate1_diagnostic_replays/game_*.json | wc -l    # must be 64
   find logs/eval/candidate1_diagnostic_replays -name 'game_*.json' | sort | \
     xargs shasum -a 256 | shasum -a 256
   # must equal the frozen digest above
   ```

   **What this proves, stated accurately:** the bytes analyzed are the bytes this
   authorization was signed over. It does **not** retroactively prove those bytes came
   from the Candidate 1 run — no per-file hashes were captured at write time, so no
   evidence exists that could establish that after the fact. A digest computed just
   before analysis would fingerprint whatever happened to be on disk; freezing it before
   signature is what makes the comparison meaningful. The provenance link to Candidate 1
   rests on the run's own artifact (commit `d2aaf4f`, checkpoint `209cf2d4…`) and on
   these files not having been touched since.

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
bash -c '.venv/bin/python -m scripts.GPU.alphazero.readout_preflight \
  --replay-glob "logs/eval/candidate1_diagnostic_replays/game_*.json" \
  --agent-id candidate \
  --opening-temp-plies 20 \
  --output logs/eval/candidate1_preflight.json
rc=$?
printf "%s\n" "$rc" > logs/eval/candidate1_preflight.exit
exit "$rc"'
```

The wrapper ends with `exit "$rc"` so the block's own status is the analyzer's. Without
it the status would be `printf`'s — normally zero — and a failed analysis would look
successful. (`rc`, not `status`: `status` is read-only in zsh.)

`--agent-id candidate` selects Candidate 1's **argmax** agent: the run bound
`candidate → argmax` and `control → tournament`, so this is the frozen post-opening
argmax-agent population.

This is seconds of work on 64 files, so it runs in the foreground.

### The exit code alone does not carry the disposition

| exit | meaning |
|---:|---|
| `0` | every frozen gate passed |
| `2` | at least one gate failed — **which one decides the disposition; see below** |
| anything else | **§A integrity abort** — the analyzer raised; there is no report at all |

**`2` is not "Candidate 2 closed".** It covers two dispositions that must never be
conflated, and they are distinguished by `gates.failed_gates`, not by the code:

| `failed_gates` contains | disposition |
|---|---|
| `undefined_q` or `empty_population` | **NO VERDICT.** Integrity stop. |
| only `override_rate_floor` / `override_rate_ceiling` / `single_game_concentration` | **Candidate 2 closed.** |

**A no-verdict stop takes PRECEDENCE**, even when a rate or concentration gate also
fired. Corrupt telemetry or an empty population means the analyzed set cannot support any
statement about the candidate — including a negative one. Halt, report no candidate
verdict, repair the cause, and obtain a fresh authorization. Do not read a co-occurring
rate failure as a closure.

Determine it mechanically rather than by eye:

```bash
.venv/bin/python -c "
import json, sys
g = json.load(open('logs/eval/candidate1_preflight.json'))['gates']
no_verdict = {'undefined_q', 'empty_population'} & set(g['failed_gates'])
if no_verdict:   print('NO VERDICT - integrity stop:', sorted(no_verdict)); sys.exit(3)
if g['passed']:  print('PASS'); sys.exit(0)
print('CANDIDATE 2 CLOSED:', g['failed_gates']); sys.exit(2)
"
echo "disposition exit = $?"
```

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

**Candidate-close gates** — these, and only these, close Candidate 2:

| statistic | closes when | reason |
|---|---|---|
| override rate | `< 0.5%` | insufficient reach to justify the spend |
| override rate | `> 15%` | excessive scope — not the conservative occasional rule hypothesized |
| share of overrides in one game | `> 50%` | concentration — the rate describes one abnormal trajectory |

**No-verdict stops** — these produce **no statement about Candidate 2 at all**, and take
precedence over any candidate-close gate that fires alongside them:

| condition | disposition |
|---|---|
| rows with corrupt Q on a visited child (`undefined_q`) | telemetry defect — halt, repair, re-authorize |
| empty population (`empty_population`) | nothing was measured — no basis for any verdict |

**Descriptive only, never a gate:**

| statistic | |
|---|---|
| **colour split** | **may never close or open the candidate** |

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

**If a NO-VERDICT stop fired (`undefined_q` or `empty_population`, alone or alongside
anything else):** record that the preflight produced **no candidate verdict**, and say so
in exactly those words. Do not record Candidate 2 as closed, and do not report a
co-occurring rate failure as if it decided anything. Repair the telemetry or population
defect, then obtain a fresh authorization. This outcome is a **tooling or data fault**,
not evidence about the candidate.

**If a candidate-close gate fired and no no-verdict stop did:** record the result in the
experiment ledger and **close Candidate 2 without spending any GPU time**. That is a
successful outcome of the preflight, not a disappointment — it is the mechanism working
as designed.

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
