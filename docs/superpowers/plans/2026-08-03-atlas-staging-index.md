# Convergence Atlas — Staging Index

**Spec:** `docs/superpowers/specs/2026-08-03-convergence-atlas-design.md`, **§3–§12
EXECUTION-FROZEN 2026-08-03.** This index freezes stage boundaries, dependencies and
handoff artifacts. It invents no downstream interfaces: **stages 2–5 are planned only
after their predecessor's interfaces exist and qualify.**

## The operator boundary — standing, and separate

Every stage below builds and qualifies tooling. **None of them authorizes the fresh
reservoir generation or any GPU measurement run.** Those require a later, explicit
"launch" authorization, exactly as Phase 0's preflight did.

A stage is complete when its tooling is built, tested, and qualified. Producing atlas
evidence is not part of any stage.

## Stages

| # | Stage | Depends on | Handoff artifact |
|---|---|---|---|
| 1 | Observer surfaces + byte-identical-off identity qualification | — | Qualified `on_flush_complete` / `on_select_child` hooks, the selection tracer, and a measured tracer-overhead figure |
| 2 | Corpus generation, deterministic assignment, geometry gates | 1 | A generator, a matching solver with min-cut witness, and the pilot geometry gate — **run against no production reservoir** |
| 3 | Warm-prefix replay + additive ladder | 1, 2 | A replay driver producing a trajectory-compounded root and the four-leg ladder, with `B`/`I`/`I+B` recorded |
| 4 | Read-outs A / B / C + artifact validation | 1, 2, 3 | The three read-outs over ladder output, plus artifact schema and authentication |
| 5 | End-to-end qualification + operator handoff | 1–4 | A qualified end-to-end chain and a written operator runbook — **no production run** |

## Why this decomposition

- **Stage 1 first because everything consumes it.** The hooks, the tracer's
  aggregation object and the identity prerequisite are the interfaces stages 3 and 4
  are written against. It is also where the scoped `mcts.py` exception is exercised
  and proven, so if byte-identity cannot be demonstrated, nothing downstream should be
  built.
- **Stage 2 before 3** because the replay driver replays *assigned* rows; assignment
  feasibility can fail (`PHASE_GEOMETRY_NO_GO`) before any replay code matters.
- **Stage 4 after 3** because the read-outs consume ladder output whose shape stage 3
  fixes.
- **Stage 5 exists** so end-to-end failures surface against real producers rather than
  surrogates — the v18 lesson that cost four contract defects and three restarts.

## Fixed boundaries

**Stage 1 owns** the observer protocols, their `mcts.py` call sites, the selection
tracer including prior-rank caching and `K(n)` evaluation for the two frozen shapes,
online counter accumulation, and identity plus timing qualification.

**Stage 1 does not own** any analysis, thresholds, or read-out logic. It produces
counters; interpreting them is stage 4.

**No stage may** change a frozen spec parameter, relax the operator boundary, or reuse
consumed v16a/v17/v18 evidence.

## Handoff rule

Each stage ends with its tooling green and its qualification recorded. The next
stage's plan is written **after** that, against the interfaces that then exist — never
against predicted ones. Writing all five up front would invent signatures the earlier
stages would then change, which is precisely how the Phase 0 plan acquired three
non-existent names (`LocalEvaluator`, `PositionRecord.search_score`,
`tests/helpers_evaluator`) that had to be corrected against real code.
