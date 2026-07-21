# FPU v2 Reservoir/Corpus Pipeline — Operator Guide (role-feasibility repair)

Covers the schema-2 (`AllocationProfile`) pipeline shipped by the
2026-07-18 role-feasibility repair plan
(`docs/superpowers/plans/2026-07-18-fpu-v2-role-feasibility-repair.md`).
Schema-1 (v1) behavior is untouched and out of scope here — see
`build_fpu_dev_corpus.py` / its own docs for that path.

**reservoir_v1 is an immutable POST-SCREEN GATE-FAIL.** Never edit, delete,
top up, reclassify, or select from
`logs/eval/fpu_v16_policy_mass_v2/reservoir_v1/` as confirmatory data — it is
discovery evidence only.

## 1. Pipeline order

```
emit-protocol → emit-gen-command → generate → qualify → screen → post-screen-qualify → select
```

- `emit-protocol` / `emit-gen-command` / `qualify` live in
  `scripts/GPU/alphazero/fpu_dev_reservoir_protocol.py` — they freeze the
  generation protocol, emit the exact generator command line, and qualify a
  GENERATED reservoir against that frozen protocol (zero-GPU: conformance +
  summary-binding-by-reconstruction + the geometric preflight). `qualify`
  never launches generation.
- `generate` is the GPU game-generation run itself (the command
  `emit-gen-command` printed).
- `screen` / `post-screen-qualify` / `select` live in
  `scripts/GPU/alphazero/fpu_dev_corpus_v2.py` (`--mode screen` /
  `--mode post-screen-qualify` / `--mode select`). `screen` (operator;
  evaluator+MCTS) screens every proposal against the cheap collision/raw-policy
  filters and the 400-sim fpu-off anchor, persisting every outcome — never
  stopping early. `post-screen-qualify` (PURE, no evaluator) writes the
  immutable PASS/GATE_FAIL `post_screen_qualification_report` that `select`
  requires — this is STAGE 2 of the two-stage feasibility split, and PASS
  means the exact selector produced a complete dry-run witness, never
  capacity bounds alone. `select` (PURE, no evaluator) hard-matches the
  persisted screen's eleven identities and deterministically selects the
  final manifest. `screen` and `select` are NEVER the same invocation.

Two further PURE, discovery-only modes run outside the production chain,
against an already-qualified screen — see §5.

## 2. Exit codes

| Code | Meaning |
|------|---------|
| 0 | OK / PASS |
| 2 | usage / IO (malformed or missing CLI input file) |
| 3 | identity or artifact mismatch |
| 4 | GATE_FAIL |

An expected capacity/geometry failure must never traceback — it always
resolves to one of these four codes with a clear message.

### The qualify PASS message (corrected)

`fpu_dev_reservoir_protocol.py`'s `qualify` subcommand prints, on exit 0:

```
reservoir geometry qualified; raw-policy role and anchor qualification remain pending
```

This replaces a plain "OK" label. Reservoir qualification and the geometric
preflight are role-agnostic (spec Sec 2) — they prove the reservoir's
GEOMETRY could support the corpus, not that any row's raw-policy `role`
(target/control) or late-target `band` is actually realizable. Only the
`screen` → `post-screen-qualify` → `select` chain later proves roles. Do not
read a `qualify` PASS as "the corpus is ready" — it only clears the
reservoir for `screen`.

## 3. Production profile (schema-2 `AllocationProfile`, 120 rows)

Verbatim JSON shape (the `PRODUCTION_PROFILE_RAW` literal,
`tests/test_fpu_v2_repair.py`):

```json
{
    "config_schema_version": 2,
    "run_kind": "production",
    "phase_allocation": {
        "target|late":       {"tuning": 40, "frozen_check": 20},
        "control|opening":   {"tuning": 10, "frozen_check": 5},
        "control|early_mid": {"tuning": 10, "frozen_check": 5},
        "control|midgame":   {"tuning": 10, "frozen_check": 5},
        "control|late":      {"tuning": 10, "frozen_check": 5}
    },
    "late_floors": {"b400_plus": 8, "b300_399": 12, "b200_299": 12},
    "late_target_band_minima": {
        "tuning":       {"b400_plus": 4, "b300_399": 8, "b200_299": 8},
        "frozen_check": {"b400_plus": 4, "b300_399": 5, "b200_299": 5}
    },
    "max_per_game": 2,
    "min_ply_gap": 12,
    "side_tol": 2,
    "corpus_size": 120
}
```

Totals: 80 tuning / 40 frozen_check, 60 target / 60 control. Targets are
required only in `late` (locked science: absolute flat/diffuse target
definition `normalized_entropy >= 0.90 AND top1_prior <= 0.025`, unchanged;
NO phase-relative quantile targets). The late-target band minima
(`b400_plus`/`b300_399`/`b200_299`) are candidates, NOT frozen, until the
exact selector produces a witness on the v1 screen (repair plan Task 14) —
if infeasible, STOP for a science decision; never silently lower a minimum.

## 4. Smoke profile (schema-2, tooling_smoke, 18 rows)

Used ONLY for the gated 400-game tooling smoke (repair plan Task 15) — see
§6 for what smoke output may and may not be used for. Verbatim from the
plan's Task 15:

```json
{
  "phase_allocation": {
    "target|late":       {"tuning": 4, "frozen_check": 2},
    "control|opening":   {"tuning": 2, "frozen_check": 1},
    "control|early_mid": {"tuning": 2, "frozen_check": 1},
    "control|midgame":   {"tuning": 2, "frozen_check": 1},
    "control|late":      {"tuning": 2, "frozen_check": 1}
  },
  "late_floors": {},
  "late_target_band_minima": {},
  "max_per_game": 2, "min_ply_gap": 12, "side_tol": 2,
  "corpus_size": 18
}
```

Every generation knob besides this allocation and `run_kind` is copied
verbatim from the v1 reservoir protocol (same checkpoints, board 24, 400
sims, eval batch 14, stall flush 48, opening-temperature settings, max moves
280, 4 workers, replay capture) — only `protocol_version`,
`config_schema_version`, `run_kind`, `games`, `base_seed`, and artifact paths
change. `late_floors` / `late_target_band_minima` are deliberately empty:
at 18 rows there is no meaningful per-band minimum to enforce.

## 5. `run_kind`: smoke can never be production evidence

`run_kind ∈ {"production", "tooling_smoke"}`, threaded through every
schema-2 artifact (protocol, config, report, manifest meta, fingerprints).
The production diagnostic entry point (`diagnose_fpu_policy_mass.py`) hard
-rejects `run_kind="tooling_smoke"` configs.

A `tooling_smoke` run's output must NEVER:
- select a coefficient,
- pass a safety gate,
- justify a strength match, or
- enter self-play.

Smoke exists only to prove the pipeline mechanically completes
(`emit-protocol → emit-gen-command → generate → qualify → screen →
post-screen-qualify → select → verify manifest/meta/fingerprints`) on a
small, cheap game count. A smoke PASS is a technical/plumbing result, never
a scientific one.

## 6. Pooled-control semantics

Controls stay in all four phases (opening/early_mid/midgame/late) as
**phase-stratified collateral coverage feeding pooled control gates — not
four independent phase hard gates.** Do not read a single phase's control
cell in isolation as a pass/fail gate of its own; the control allocation
exists to feed the diagnostic's pooled control statistics, not to stand as
four separate phase-level acceptance criteria.

## 7. Inactive-band-gate honesty rule

`DEV_BAND_MIN_N = 20` (`diagnose_fpu_policy_mass.py:99`) is the per-stratum
activation minimum for a rate gate. When `dev_safety_verdict(...,
include_stratum_census=True)` runs (the v2 operator path, via
`--dev-corpus-config`), the verdict's metrics record, per stratum key:

- `f"{stratum_key}_stratum_sizes"` — every stratum value's target-row count.
- `f"{stratum_key}_inactive_strata"` — the sorted list of stratum values
  whose count is `< DEV_BAND_MIN_N`, i.e. whose rate gate did NOT run.

**Honesty rule:** never imply a band (or phase, if `new_collapse_stratum`
is ever `"band"` again) rate gate ran when its minimum sample size wasn't
met. A stratum in `inactive_strata` contributed no verdict signal — report
it as inactive, not as a silent pass. Default (`include_stratum_census=False`)
keeps v1 gate-JSON bytes identical.

## 8. Discovery commands

Both run against an already-qualified, immutable screen (never mutate it),
and both authenticate their input first — the full identity chain
(identities, rows-vs-meta, config re-derivation) runs before any analysis,
proving the INPUT is the qualified artifact. This does NOT make the OUTPUT
confirmatory: both reports are stamped `discovery_only`.

### `analyze-screen-feasibility`

Runs the exact qualifier+selector against an existing screen for a
candidate `AllocationProfile`, as a dry-run witness (repair plan Task 14).

```bash
.venv/bin/python -m scripts.GPU.alphazero.fpu_dev_corpus_v2 \
  --mode analyze-screen-feasibility \
  --config logs/eval/fpu_v16_policy_mass_v2/reservoir_v1/fpu_dev_corpus_v2_config.json \
  --screen logs/eval/fpu_v16_policy_mass_v2/reservoir_v1/fpu_dev_source_screen.csv \
  --profile-json logs/eval/fpu_v16_policy_mass_v2/analysis/production_profile.json \
  --out logs/eval/fpu_v16_policy_mass_v2/analysis/production_feasibility.json
```

`--config` is the screen's OWN config (authenticates the input); `--profile-json`
is an `AllocationProfile` JSON (the `PRODUCTION_PROFILE_RAW` shape) plus a
`selection_seed` int. Exit 0 + `status=PASS` means the profile's exact
selection witnessed on this screen; exit 4 + `status=GATE_FAIL` names the
binding constraint (e.g. `target|opening capacity 0 < demand 45`).

### `sizing-analysis`

Deterministic whole-game resampling over an existing screen, to justify a
production game count via finite-reservoir subsampling (repair plan Task 16).

```bash
.venv/bin/python -m scripts.GPU.alphazero.fpu_dev_corpus_v2 \
  --mode sizing-analysis \
  --config logs/eval/fpu_v16_policy_mass_v2/reservoir_v1/fpu_dev_corpus_v2_config.json \
  --screen logs/eval/fpu_v16_policy_mass_v2/reservoir_v1/fpu_dev_source_screen.csv \
  --profile-json logs/eval/fpu_v16_policy_mass_v2/analysis/production_profile.json \
  --game-counts 1200,1800,2400,3000,3600,4200,4800 --trials 299 --seed 20260718 \
  --out logs/eval/fpu_v16_policy_mass_v2/analysis/sizing_report.json
```

`--trials 299` is the preregistered all-success requirement for the exact
one-sided 95% binomial lower bound `>= 0.99` (run `--trials 50` once first
to measure runtime). Freeze rule: production game count = the next larger
tested tier above the smallest tier with `meets_criterion` true; if the
smallest qualifying tier is the largest tested tier, use that tier; if no
tier qualifies, STOP — going beyond the discovery screen is a user decision.
This is finite-reservoir subsampling of THIS reservoir, not a fresh one.

## 9. §2 correction: the policy-mass / concentrated-openings note

"policy-mass ≈ absolute in concentrated openings" is a **hypothesis**
requiring `Q_parent ≈ 0`, not a measured fact. The context-relative
policy-mass FPU rule is `FPU = Q_parent − r·sqrt(P_explored)`; near
`Q_parent ≈ 0` the reduction's absolute size approaches what an
absolute-constant rule would have produced at that node, which is why an
opening position — where the parent value is often close to neutral — was
informally described as a case where "policy-mass behaves like absolute."
That description is a mechanism-level conjecture about the FORMULA's
behavior under a `Q_parent ≈ 0` precondition. It has NOT been measured
against real opening-phase rows on this project's checkpoints/board, and
must not be cited as an established result in any qualification report,
gate decision, or operator summary. Treat it as a note to test for, not a
premise to build on.

## 10. Smoke run record (2026-07-19) — technical PASS, not a scientific result

The 400-game 24×24 `tooling_smoke` chain ran end-to-end from the frozen protocol with no manual artifact edits:

- `emit-protocol` → `emit-gen-command` → generate (400 games, base_seed 20280000, seed range disjoint from v1) → `qualify` exit 0 → `screen` (2,032 proposals, 581 kept) → `post-screen-qualify` **PASS** exit 0 → `select` exit 0.
- Manifest: 18 rows — 12 tuning / 6 frozen_check, 6 target / 12 control, sides 10 black / 8 red (within `side_tol 2`); `run_kind=tooling_smoke` in the protocol, config, report, and manifest meta.
- Idempotency: re-running `post-screen-qualify` and `select` was a clean accept with byte-identical artifacts.
- Isolation: the production diagnostic (`require_production_run_kind`) rejects the smoke config with the §5 SystemExit.
- Artifact SHA-1s: protocol `ee43e847…5065`, post-screen report `040ccce…b2c`, manifest `fbc87b1…d01`.

**This proves plumbing only.** The smoke deliberately omits production band floors; its outputs must never select a coefficient, pass a safety gate, justify a strength match, or enter self-play.

## 11. Selector v2 (constraint-aware split assignment) + revised sizing + smoke_v2 (2026-07-21)

- **Selector v2** (`split_assignment_version: 2`, commits a005fa0 + 08ef42d): schema-2 selection pre-pins scarce per-split-minima-band games to splits (deterministic largest-shortfall-first cover), with the historical pin-free assigner retained as fallback — the feasible set provably contains selector v1's (differential fuzz: 250 pools × 30 seeds, zero regressions). Schema-1 selection is byte-identical. Motivation: the 8-attempt greedy caused false infeasibility (measured 255/299 → 288/299 at 4,200 games with more attempts).
- **Sizing (selector v2, preregistered ladder incl. 4,400/4,600):** `sizing_report_selector_v2.json` (SHA-1 92c… superseded; final 81499afc2ac5c04300a4c7c5f376cf8b65f7335f) — **4,600 games: 299/299, lb95 0.9900 ≥ 0.99 MEETS**; preregistered next-tier-up rule → **production count 4,800**. Not emitted; production launch requires explicit authorization.
- **smoke_v2** (400 games, seed 20290000, selector-v2 tree): full chain; **controlled sampling GATE_FAIL** at post-screen-qualify (`target|late capacity 5 < demand 6` — 6 kept late-target rows in 3 games under ≤2/game), select refused exit 3, idempotent, isolation verified. Per the standing rule this is a small-sample yield result, never permission to top up or weaken production minima. Failure-machinery plumbing thereby exercised end-to-end on real data.
- **Assigner exercise (zero-GPU):** `analyze-screen-feasibility` on the authenticated smoke_v2 screen with a discovery-only tooling profile (target|late 2+2, per-split b200_299 minima) — PASS, witness 16 rows, `split_assignment_version: 2`, pins distributed the scarce-band games across splits on attempt 0. Plumbing evidence only (`smoke_v2_assigner_exercise.json`, SHA-1 73a337f1…).

## 12. Scientific amendment `b400-coverage-floor-v1` (2026-07-21, user decision)

**Amendment:** the b400_plus late-target floor is reduced from total 8 (per-split 4 tuning / 4 frozen_check) to **total 4 (per-split 2 / 2)**. Everything else is unchanged: 99% reliability criterion (exact one-sided lb95 ≥ 0.99, 299 all-success trials), 60/60 target/control allocation, absolute target definition, b300_399 and b200_299 minima, side balance, spacing, per-game cap, whole-game split isolation.

**Rationale (user):** b400_plus has fewer than `DEV_BAND_MIN_N = 20` cases, so it can never activate an independent per-band safety gate — it is qualitative coverage, not standalone statistical power. Two examples per split preserve that coverage without pretending otherwise. Exploratory sensitivity (read-only): 4+4 first passes at 4,600; 3+3 at 4,400 (saves only 200 games); 2+2 at 3,600; 1+1 no improvement over 2+2 — so 2+2 is the boundary. The two independent 400-game smokes (800 pilot games, 2 b400 rows) already confirm the discovery-rate yield; no further pilot needed.

**Frozen profile:** `logs/eval/fpu_v16_policy_mass_v2/analysis/production_profile_v2_b400amend.json`.
**Preregistered BEFORE the decision sizing run:** ladder `3000,3200,3400,3600,3800,4000`, 299 trials, seed 20260718, criterion unchanged, production count = next larger tested tier above the smallest qualifying tier. Generation launches only on explicit user authorization after the sizing evidence exists.

## 13. Production authorization record (2026-07-21)

> Production reservoir authorized at 4,000 games, board 24, four workers, seed range `[20300000,20304000)`, `run_kind=production`, and no top-up. This intentionally deviates from the preregistered 3,800 margin result because 4,000 independently met the unchanged criterion at 299/299, while 3,800 produced 298/299. All allocation, target, b300/b200, side, spacing, isolation, and amended b400 2+2 requirements remain unchanged.

Decision basis: `sizing_report_b400amend.json` (SHA-1 `7ff76fcb8a5a9d58f1ca2227767142a3c2f307dd`) under amendment `b400-coverage-floor-v1` (§12) with the frozen profile `production_profile_v2_b400amend.json` (SHA-1 `eb1dd21648e49388bff92de8c7831eb0a1a3f6e8`).
