# Convergence Atlas — Closeout

**Closed:** 2026-08-05 · **Status:** CLOSED. Progressive widening is closed with it.
**No further atlas GPU work is authorized.** Shipped search is unchanged.

The atlas ran its authorized pilot and returned two stopping findings. This document
records them, states precisely what is **unanswered** rather than failed, and marks the
pivot away from tree-local search heuristics.

## Hypotheses and reasoning

The atlas was a read-only diagnostic, not a candidate search version. It responded to
two facts that the preceding search-reliability line had never tested directly:

1. **The candidate set was the last untouched tree-local axis.** Shipped MCTS exposes
   every legal move at every node. `c_puct`, three FPU formulations and v18 changed
   values or backups; none restricted which moves were eligible for selection.
2. **The collateral gates had rejected four interventions without a strength match.**
   They had never been calibrated against whether a change moved 400-simulation search
   toward a stable higher-budget result.

Phase 0 then established that deployment search is warm, not fresh: tree reuse is
additive, so a nominal 400-simulation search ends at `I + 400` visits. The atlas
therefore replayed the complete game prefix and measured one warm-root ladder at
nominal budgets 400, 1,600, 3,200 and 6,400.

The frozen hypotheses were:

- **Read-out A:** features available at the batch-safe 320-completion boundary can
  predict which ordinary 400-simulation searches disagree with a stable 3,200/6,400
  reference.
- **Read-out B:** the old collateral gates can be calibrated by asking whether they
  reject changes that consistently move toward the stable higher-budget reference.
- **Read-out C:** a small prior-ranked progressive-widening rule can affect misleading
  roots more often than stable ones while retaining the stable root moves and replies.

## Preregistered expected outcomes

| Result | Frozen consequence |
|---|---|
| Detector passes; widening fails | A bounded 320+80 verification prototype becomes eligible. |
| Widening passes; detector fails | A small progressive-widening prototype becomes eligible. |
| Both pass | Compare separation and implementation risk; neither wins by default. |
| An old gate is marked `needs review` | Redesign and freeze that gate before it judges a prototype. |
| Neither tree-local method separates misleading from stable roots | Close tree-local heuristics and pivot to direct strength or a separately designed distillation project. |
| Corpus capacity fails | Record an operational no-go; do not claim that the information cannot exist. |
| Both widening shapes genuinely fail the authoritative pilot check | Close progressive widening without adding or tuning another shape. |

The last two paths were deliberate terminal outcomes, not software failures or
incomplete runs. They are the paths the valid pilot took.

## What ran

| | |
|---|---|
| qualified code | `1332bcc`, suite 2,636 passed / 4 skipped / 53 deselected / 0 failed |
| bound HEAD | `2484786` (docs-only ahead of the qualification commit) |
| checkpoint | `alphazero-v2-calib020-from0409/model_iter_0001.safetensors`, `209cf2d4…` |
| corpus | `[20321000, 20321024)` — 24 games, pilot interval only |
| sampling seed | `20260806` · phase rule: Amendment 5 |
| sequence | preflight → generate → preflight `--pilot-dir` → gate **PASS 24/24** → `run-pilot` **OK**, 24/24 measured, no failed rows |
| wall clock | **5m08s** for the 24-row ladder (660 prefix plies + 24 × 6,400 simulations) |

Amendment 5 did what it was written to do: the same shape of corpus that produced 18/24
with both late cells empty under absolute bounds produced **24/24 with nothing unmet**.
Game lengths were 37–83 (median 59) against the retired run's 39–76, so the corpus did
not change character — only the definition of "late" did.

## Stopping finding 1 — `PROJECTED_CAPACITY_NO_GO`

```text
p_m 0.4583  (11/24 misleading: 8 by move, 6 by value)
p_s 0.0417  ( 1/24 stable-negative)
required N = max(60/p_m, 75/p_s) = max(131, 1800) = 1800  >  frozen maximum 400
class counts   misleading 11 · stable_negative 1 · ambiguous 0 · no_stable_reference 12
```

**The binding constraint is stable-negative scarcity, not misleading scarcity.**
Misleading positions were abundant and on their own would have needed `N ≈ 131`. One
stable-negative row in 24 drove the requirement to 4.5× the ceiling.

The deeper number is that **12 of 24 rows had no stable deep reference at all** — 3,200
and 6,400 did not agree on move, value within `0.10`, and a top-two margin `≥ 0.05`.
Half the corpus was unclassifiable.

§3's rule is to stop with a projected capacity no-go rather than spend the full run on a
design already known to be underpowered. That is what happened.

## Stopping finding 2 — progressive widening: `both_fail`

Both frozen shapes returned a genuine `FAIL` under the frozen three-way precedence —
not `INCONCLUSIVE`, which is the distinction that precedence exists to draw.

| shape | root retention (floor 0.95) | depth-1 retention (floor 0.90) |
|---|---|---|
| `c4a05` (C=4, α=0.5) | **1.0** | **0.6842** |
| `c13a03` (C=13, α=0.3) | **1.0** | **0.6842** |

**Retention alone rejects both shapes, and that is the whole basis for this finding.**
Depth-1 retention missed the floor by a wide margin, identically for both shapes, over
**12 stable-reference-eligible rows**. Root retention was perfect, so the failure is
specific and legible: widening would keep every stable root move and drop roughly a
third of the depth-1 replies that stable deeper search requires.

**The intervention numbers are NOT the basis, and should not be quoted as if they were.**
Their denominators are sparse and asymmetric — `c4a05` had 1 misleading row scored and
**17 inconclusive**, `c13a03` had 10 scored and 2 inconclusive, and both had a single
stable-negative row. That asymmetry is the `K(n+14)` lag bound behaving differently
across the two shapes, exactly as §8 intends, but one-row denominators carry no weight.
The rejection does not need them.

This finding **stands independently of finding 1**, because §8 preregistered the pilot's
early static widening check as authoritative on the pilot alone, before any of this ran.
Per §8: close progressive widening **without inventing another shape.**

## What is UNANSWERED — not failed

- **Read-out A's selectivity.** Whether 320-prefix features predict which 400-simulation
  searches disagree with a stable deep reference was never tested. The corpus could not
  supply the classes.
- **Read-out B's gate calibration.** Whether the inherited collateral gates fire on
  changes that move toward the stable deeper reference is likewise untested.

Both remain open questions. Nothing here is evidence that the information does not
exist — this is an operational capacity failure, and the v18 closeout's framing is
preserved deliberately.

## What the remaining-budget result does and does not establish

```text
median_remaining 66 · quartiles 51.5 / 66 / 77 · zero_budget_fraction 0.0 · DEPLOYABLE
```

§4 preregistered that a **median of zero fails** Read-out A's controller-deployability
claim. It is 66, with no zero-budget rows, so a bounded 320+80 controller is
**feasible** — there is real budget left at the batch-safe boundary.

**Feasibility is not justification.** With Read-out A's selectivity unanswered, there is
nothing for such a controller to act on. This result does **not** justify a controller
prototype or a verification-search prototype.

Also measured, and now on the record: inheritance-reset rate mean `0.024`, max `0.129`,
11 of 24 rows with any reset, inherited visits `I` median 60 (range 0–400). The warm
start is genuinely warming.

## Prohibited

Do **not**:

- enlarge the corpus, or generate beyond `[20321000, 20321024)`;
- loosen the stable-reference criteria to manufacture classifiable rows;
- add, tune or interpolate a third widening shape;
- reuse these 24 rows, or the permanently retired `[20320000, 20320024)` rows, as fresh
  evidence;
- change shipped search.

The 12 unclassifiable rows are a **finding about this checkpoint at these budgets**, not
a labelling problem to be defined away.

## The pivot

Tree-local search heuristics are closed. Across this line: `c_puct` (falsified), three
FPU formulations (absolute, parent-relative policy-mass, baseline-preserving
policy-mass — all rejected), a depth-2 provisional backup (preflight-rejected), and now
progressive widening, the one axis that had never been touched. The candidate set was
the last untried lever, and it fails retention.

Next work should be **direct playing-strength work**, which is the only benchmark this
line ever accepted as decisive.

**Broad high-budget distillation may be considered as a new project** — but this pilot is
a warning about its premise. **Half the rows had no stable 3,200/6,400 reference.** A
distillation scheme that treats a single high-budget result as truth would be training
on a target that, here, disagreed with itself between 3,200 and 6,400 simulations half
the time. Any such project needs its own stability criterion first, and should not
inherit the assumption that "deeper is truth".

## What survives as durable tooling

The atlas apparatus is qualified, general, and unspent: corpus geometry with min-cut
witnesses, the reservoir producer with fail-closed provenance, warm-prefix replay, the
additive ladder with its batch-safe boundary, the selection tracer, the three read-outs,
the authenticated artifact with its reloadable inverse, and a two-phase operator runbook
with measured provenance and exit-status sidecars. Any successor corpus study can reuse
it without rebuilding measurement infrastructure.

## Authorization state

Both pilot authorizations are **spent**. Continuation generation, `run-final`, re-runs,
top-ups and replacements are unauthorized and remain so. **No further atlas GPU work is
authorized.**
