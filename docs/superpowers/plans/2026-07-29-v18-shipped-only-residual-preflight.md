# v18 Shipped-Only Residual Preflight Implementation Plan (revision 24)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the read-only tooling, the verified non-A control pool, and the frozen numeric criteria needed to decide — from shipped-search measurements alone — whether the v18 depth-2 provisional-backup mechanism is worth implementing, without editing `mcts.py` or running a single positive-cap search.

**Architecture:** Seven new pure-Python modules plus one narrow parameterization of an existing capture script. The formula helper is written once here and later imported by `mcts.py`, satisfying the spec's one-implementation rule. A read-only tree walker derives every v18 metric from a finished search tree, so no hot-path instrumentation exists anywhere. A crossover module computes the static first-order selection analysis. A control-pool module establishes and freezes a genuinely non-A discovery population with proven exclusions. A criteria module freezes every numeric threshold and decision rule. Measurement and judgement live in separate modules, and **all tooling is committed at a clean HEAD before any measurement runs**, so no implementation choice can be influenced by a result.

**Tech Stack:** Python 3, MLX (GPU eval only, in the execution phase), pytest, existing `scripts/GPU/alphazero/` modules (`mcts.py` read-only, `fpu_provenance.py`, `fpu_state_hash.py`, `fpu_dev_reservoir_protocol.canonical_json_bytes`, `eval_runner.py`, `position_probe_cases.py`).

**Source spec:** `docs/superpowers/specs/2026-07-29-v18-depth2-provisional-backup-design.md` (revision 3, approved for the preflight step only).

## Revision 2 change log

Revision 1 was reviewed and rejected. Nine findings, all incorporated:

1. **Control population was degenerate.** The proposed source
   `logs/eval/calib020_post_opening_sweep/position_probe_cases.csv` holds 240 rows
   over only **30 distinct positions repeated across 8 checkpoints**, and those 30
   positions *are* gate C. Canonical exclusion of C left zero controls. **New Task 4**
   establishes and freezes a verified non-A pool.
2. **First-past-the-post still let selected-A choose the selector.** Replaced with a
   **single formula frozen before measurement** (`contribution_weighted_positive_mass`);
   the other two are descriptive only and cannot rescue a failure. A control-only
   quantile rule now supplies the row-selection cutoff the plan previously omitted.
3. **Task 5 did not implement A/6,400 authentication.** It authenticated against the
   400-sim values, which 6,400 must fail. Now authenticates against
   `logs/eval/v15_budget_check/a_predrop_base_6400sims.csv/position_probe_cases.csv`
   (SHA-1 `a17d4737c747e2799253bebbc3d0261e0e697114`) on exact case set, per-case value
   and top share, with two byte-identical captures.
4. **Execution order defeated the measurement/judgement split.** All tooling is now
   implemented and committed before any measurement; a separate execution phase runs at
   a clean final HEAD.
5. **Criteria artifact was emitted before its own commit.** Emission moved to the
   execution phase at a clean HEAD, with the SHA independently re-derived from the
   committed module and byte-compared rather than trusted from an adjacent file.
6. **Selector sizing was omitted while still producing a final PASS.** **New Task 8**
   adds screen telemetry, selector predicates and sizing; the verdict is
   `MECHANISM_PREFLIGHT_PROVISIONAL_PASS` until Task 8 passes.
7. **Crossover delta must be signed.** Negative residuals can *increase* reply scanning;
   a negative prediction is a meaningful result, not invalid input.
8. **Task 1's observed-A example had the parent perspective reversed** (passed `-0.087`
   where the parent-to-move value is `+0.087`), contradicting Tasks 2/3 and the spec.
9. **Several criteria lacked exact formulas or denominators.** All now mechanical.

## Revision 3 change log

Revision 2 was reviewed and rejected. Eight further findings, all incorporated:

10. **Task 4 could not build its pool without search.** Replay moves carry only
    `col, n_legal, player, ply, root_top1_share, root_total_visits, root_value,
    row, selected_visit_count, selected_visit_rank` — no tree, no depth-2
    population, no residuals — and `root_value` belongs to whichever checkpoint
    held that colour, not consistently `calib020_0001`. Task 4 now freezes a
    broad **source universe** from immutable replay predicates only; the cohort
    is chosen after measurement by a **precommitted matcher** (new Task 4b).
11. **Task 8 had no data path.** It consumed only the Task 7 artifact, which
    holds A plus the final controls — not a whole-reservoir census. Task 7 now
    measures a **per-game census** over the universe, serving both the matcher
    and sizing from one pass.
12. **Selector predicates were unfrozen.** Exact numbers for near-even, minimum
    eligible leaves, flip-control exposure (operator and both constants),
    matching variables and tolerances, phase/ply windows, per-game caps and
    spacing are now in Task 5. `test_no_criterion_uses_absolute_residuals` is
    renamed — flip controls legitimately use absolute exposure; the prohibition
    binds **target** selection only.
13. **`R_min` could reject the mechanism for being selective.** Small reply
    reduction on ordinary controls is desirable, and weak-cap under-reach must
    advance, not reject. `R_min` now derives from the **prospective-target
    subset** across all caps, failing only if no cap predicts positive
    conversion.
14. **Completed-but-failing sizing must be `PREFLIGHT_FAIL`**, distinct from
    missing sizing, or an infeasible selector never formally rejects v18.
15. **A/6,400 authentication was caller-controlled**, and adding fields
    unconditionally would have changed the v17 default bytes. The mode now
    self-constrains and the schema is version-dispatched.
16. **Game identity regressed a v17 lesson.** `(replay_dir, game_idx)` misses a
    renumbered copy. Now uses replay-content SHA-1 via the established
    `diagnose_fpu_baseline_policy_mass.game_identities`, with **at most one
    control position per game** so no clustered bootstrap is needed.
17. **The `mcts.py` no-change check could never pass.** This branch inherits
    v17's changes — `git diff main..HEAD -- mcts.py` is 62 lines. New **Task 0**
    pins the branch point and the file's content SHA-1, and every later check
    compares against those.

## Revision 4 change log

Revision 3 was reviewed and rejected for execution. Eight findings plus one
hardening, all incorporated:

18. **Task 5 needed information that could not exist yet.** The AUC threshold was
    to be approved against the cohort's actual `n_C`, but the cohort is not built
    until after Task 7's census. **Matching cardinality is now frozen at 1:1,
    `n_A = n_C = 30`**, so the bootstrap operating-characteristic table is
    computable and approvable before any measurement. Failing to fill all 30 is a
    preflight failure, not a smaller cohort.
19. **The matcher was underspecified.** "Greedy nearest-neighbour" can fail on
    A-row ordering even when a complete valid matching exists, and the ratio was
    never stated. Replaced with **deterministic minimum-cost bipartite matching**
    at frozen cardinality, refusing unless a complete matching exists.
20. **The matched cohort was not an artifact.** Task 4b now canonically emits and
    binds the matched rows, the full report, and every input SHA-1; Tasks 8 and 9
    authenticate and consume it explicitly.
21. **The census schema was incomplete.** A canonical `census_positions.csv` now
    carries every field the matcher and the sizing join require.
22. **The two open numeric blocks are frozen** — census-per-game and the sizing
    ladder — and the deterministic-prefix/resampling tension is resolved.
23. **`R_min`'s population was ~3 rows.** Adopted the recommended split: the
    matched cohort supplies AUC and the exposure cutoff; the **broad non-A
    census** supplies prospective-target `R_min` and revisit-form calculations.
24. **Three test sketches still reflected revision 2** — `freeze_source_universe`
    arguments, capture six-field identity, and the obsolete `R_min` reason name.
25. **Task 8 weakened the flip predicate** to "and/or"; it now imports the Task 5
    predicate rather than restating it.
26. **Hardening: separate frozen base seeds** for selected-A and the census, since
    `game_idx` is reservoir-local and can collide across sources.

## Revision 5 change log

Revision 4 was reviewed and rejected for execution. Five findings plus four
corrections, all incorporated:

27. **The matcher's matrix did not enforce one control per game.** Reducing to
    the best position per game *per A row* still leaves two positions from one
    game as two distinct columns, so two A rows could take both. Columns are now
    **games**, cell `(a, g)` holding game `g`'s best admissible position for A
    row `a`; the invariant lives in the matrix shape. Three new tests, including
    brute-force agreement on small matrices.
28. **Six global ply quantiles could miss a phase entirely.** The census is now
    **phase-stratified 1/1/1/3** (opening, early_mid, midgame, late×3) with exact
    nearest-rank positions without replacement; missing phases contribute zero
    and are reported.
29. **Tier 800 is not 299 independent trials.** Drawing 800 from 800 returns the
    same set every time. Tiers 200–700 qualify probabilistically; **800 is a
    single degenerate witness** usable as the next-tier-up operational size; if
    700 fails, sizing fails. The pass rule now imports
    `fpu_dev_corpus_v2._binomial_lower_bound` (exact **one-sided**
    Clopper–Pearson, `alpha=0.05`) instead of saying "95% interval", and the
    reported witness comes from a **successful trial**, not the content-SHA
    prefix, which can fail even when all trials pass.
30. **The AUC simulation had no data-generating model.** Frozen: equal-variance
    Gaussian location shift `δ = √2 Φ⁻¹(AUC)`, 2,000 outer datasets, separate
    outer/bootstrap seeds, the protocol's percentile convention, tie handling,
    and Monte Carlo standard errors on every estimate — labelled model-specific,
    not distribution-free.
31. **The sizing universe is now exactly 800 games** after exclusions, by
    content-SHA order, before measurement, whichever source wins. Fewer than 800
    eligible games stops. This keeps the ladder valid and caps the census at
    4,800 searches.
32. Task 8's interface gains the matched-cohort artifact; Task 7's "control-pool"
    wording becomes the universe record; the seed audit compares **complete
    derived seed sets** rather than bases; and Task 4 Step 5 chooses the source
    while the artifact is frozen at Execution step 3 on the clean HEAD.

## Revision 6 change log

Revision 5 was reviewed and rejected for execution. Seven findings plus two
closures, all incorporated:

33. **Task order contradicted module dependencies.** Tasks 4 and 4b import from
    the criteria module that Task 5 creates. Execution order is now
    `0 -> 1 -> 2 -> 3 -> 5 -> 4 -> 4b -> 6 -> 7 -> 8 -> 9`, with object-identity
    tests proving the constants are imported, not restated.
34. **`sizing_analysis_core` is not role-neutral.** It hard-calls
    `post_screen_qualification_report` and `sample_v2_rows` and iterates
    `late_target_bands` — all v2 two-role specific. Task 8 now **modifies**
    `fpu_dev_corpus_v2.py` with a narrow version-dispatched hook, preserving the
    default path byte-for-byte, rejecting v18 roles under v16/v17 schemas, and
    carrying real v17 producer byte-regression tests. Only
    `_binomial_lower_bound` is imported unchanged.
35. **Four-role assignment was undefined and the predicates overlap.** The roles
    are now an **exclusive classification** with a frozen assignment order
    (identity → flip → target → representative), a stated overlap resolution,
    representative selection ordered by a hash key so residual-independence is
    structural, and explicit refusal. The mutual-exclusivity test is replaced by
    a totality-and-partition test.
36. **The bootstrap "lower bound" named the upper percentile.** Rank
    `0.95 * (n - 1)` is the 95th percentile. Now the **one-sided** 95% lower
    bound at `q = 0.05`, matching the sizing convention, with a test that fails
    if the upper quantile is substituted.
37. **Exact-800 membership must retain zero-yield games.** The five-step order is
    frozen: authenticate → drop forbidden whole games → content-SHA sort and take
    800 (fixing `all_game_ids`) → enumerate census inside them → apply position
    exclusions **without** removing games from `all_game_ids`. Ten behavioural
    tests added, previously absent.
38. **Single-slot phase quantile specified** as `q = 1/2` at index `ceil(q*n)`
    over ascending distinct plies; "ties to the lower ply" is withdrawn as it
    described a different algorithm.
39. **A seed collision is a pre-search STOP**, not a base change — by Execution
    step 3 the base is committed and embedded in the criteria. Remediation
    requires an amendment, tests, commit and restart from step 1, with a
    zero-evaluator-call refusal test.
40. **Closures:** an undersized broad prospective-target subset (floor 16 rows)
    is `PREFLIGHT_FAIL` rather than a `0/0` division; and both A/6,400 capture
    SHA-1s are recorded in the bundle and the verdict so Stage 0 can bind the
    accepted reference.

## Revision 7 change log

Revision 6 was reviewed and rejected for execution. Four findings plus two
mechanical corrections, all incorporated:

41. **Representative selection was still residual-conditioned.** Assigning them
    last, after identity/flip/target had each removed rows on residual criteria,
    then hash-sorting the survivors, does not make the CANDIDATE SET
    residual-independent. The order is now
    `target -> representative -> {identity, flip}`, with targets carrying an
    explicit `NOT flip_control` term to preserve flip priority, and
    representatives drawn from non-target rows before identity or flip
    eligibility is inspected at all. Shortfall stops; representatives are never
    revisited.
42. **The v2 sizing hook route is frozen** as schema-dispatched role vocabulary
    and allocation, reusing the existing `post_screen_qualification_report` and
    `sample_v2_rows`, with late-target-band reporting made optional. The
    "callbacks or schema dispatch" ambiguity is gone, and
    `v18_selector_sizing.py` explicitly contains no selection algorithm.
43. **The A/6,400 captures are wired into Task 9.** A new authenticated
    `a6400_reference_bundle.json` carries both paths and SHA-1s, the
    byte-identity result, the historical source SHA-1 and the per-case
    authentication result; `evaluate(...)` takes it as an input and refuses on
    any failure.
44. **AUC-tail tests moved to Task 5**, where the quantile and the OC generator
    are implemented — seven tests covering the `q = 0.05` tail, interpolation on
    a known vector, the delta mapping, stream separation, reproducibility, and
    that the generator reads no measurement artifact.
45. **Mechanical:** universe step 2 excludes whole games by content SHA **only**
    (canonical hashes are position exclusions at step 5, and applying them
    earlier defeats zero-yield retention); Task 8's `git add` now includes
    `fpu_dev_corpus_v2.py` and the new hook test.

## Revision 8 change log

Revision 7 was reviewed and rejected for execution. Four findings, all
incorporated:

46. *(SUPERSEDED by revision 9 items 50–51: the schema number and the
    "unmodified selector" claim were both wrong.)* **The selector schema
    contract is now fully frozen** — schema 4, split
    vocabulary `("all",)` with the reasoning recorded (spec §11 replaces the
    frozen split with a separate held-out reservoir, so "whole-game split
    isolation" is between corpora), the four-role vocabulary, the exact 40-row
    role/phase/split allocation, `late_target_bands` emitted-but-empty so
    `sizing_analysis_core:3927` needs no edit, and **resolver functions**
    (`roles_for_schema`, `splits_for_schema`, `allocation_for_schema`) mirroring
    the existing `PROFILE_RUN_KINDS_V3` precedent at `:246-252`. `_ROLES`,
    `SPLIT_ALLOC_V2`, `SPLITS` and the other legacy constants keep their values
    and types; only the hardcoded `"tuning"`/`"frozen_check"` literals inside
    `corpus_size`, `split_totals` and `quota_by_phase` become schema-driven, and
    byte-identically so for schemas 1–3.
47. *(SUPERSEDED in part by revision 9 item 52: the bundle does NOT carry its
    own SHA-1 — a file cannot contain the hash of its own bytes. The builder
    returns it and the binder records it externally.)*
    **The A/6,400 bundle has a producer and a re-deriving verifier.**
    `build_a6400_reference_bundle` in Task 6 emits it with an artifact kind,
    schema version and its own SHA-1; `load_verified_a6400_bundle` recomputes
    every claim — live capture hashes, its own byte comparison, the reopened
    historical source hash, all 30 case authentications — and rejects extra or
    missing keys and path substitution. Five attack tests added.
48. *(SUPERSEDED by revision 9 item 53: those artifacts are untracked, so the
    basis is split into a tracked fixture plus an operator-only check.)*
    **The v17 byte regression gains a pre-edit identity basis.** New Task 8
    Step 0 names the real chain (`fpu_dev_corpus_v2_config.json` +
    `fpu_dev_source_screen.csv` → `fpu_dev_corpus_v2_manifest.csv` + its
    `.meta.json`), re-runs it at the unedited HEAD, byte-compares against the
    committed outputs, and records all SHA-1s to
    `tests/golden/v18_v2_selector_pre_edit_basis.json` **before** any edit. A
    non-reproducing chain stops the task.
49b. *(Revision 8's schema-4 choice and "unmodified selector" claim were both
    wrong; see revision 9.)*

49. **Three stale contracts fixed:** Task 8's test comment now states the
    `target -> representative -> {identity, flip}` order (plus a test that
    representatives are chosen before identity and flip exist); Task 8's Step 3
    prose drops the retired callbacks route for schema dispatch; and Task 9's
    `Produces` signature carries `a_reference_bundle`.

## Revision 9 change log

Revision 8 was reviewed and rejected for execution. Three P1s and one P2, all
incorporated. Two were factual errors in revision 8:

50. **Schema 4 is already v17's.** `parse_allocation_profile` accepts `(2, 3, 4)`
    at `:337` and the live v17 development config is `config_schema_version: 4`.
    Revision 8's "current maximum is 3" came from reading the
    `PROFILE_RUN_KINDS_V3` comment instead of the parser. **v18 is schema 5**;
    reusing 4 would collide with authenticated v17 artifacts.
51. **`sample_v2_rows` cannot run unmodified under one split.** Beyond the three
    `AllocationProfile` properties, the two-split assumption is also in
    allocation parsing and totals (`:381`), the two-way assignment itself
    (`:818`, `u_t, u_f = realizable("tuning"), realizable("frozen_check")`), and
    the fill/side-balance/witness loops over module-level `SPLITS` (`:1240`).
    The frozen route now keeps schemas 1–4 on the current two-way algorithm,
    gives schema 5 a deterministic one-split assignment, routes every loop
    through `alloc.splits`, reads late-band handling from the profile rather
    than `LATE_TARGET_CELL`, and mutation-pins both directions.
52. **The bundle producer now lives in Task 6**, with its files, interface and
    the five attack tests — revision 8 described it only from Task 9's side, so
    no task owned it. And the bundle **no longer claims its own digest**, which
    is impossible without recursion: the builder returns the SHA-1 of the bytes
    it wrote, and the binder records it externally.
53. **The pre-edit basis cannot rest on ignored files.** All four v17 artifacts
    are untracked under `logs/*`, so calling them "committed" was wrong. Split
    into (a) a tracked portable fixture under
    `tests/golden/v18_v2_selector_pre_edit_basis/` that the suite enforces, and
    (b) a one-time operator-only reproduction against the real ignored
    artifacts whose result is recorded rather than tested.

## Revision 10 change log

Revision 9 was reviewed and rejected for execution. Three findings plus one
stale marker, all incorporated:

54. **Schema 5's authority was described but not enforced.** `alloc.splits` was
    used without being defined, and `allocation_for_schema(5)` was a producer
    default only — so a schema-5 config could shift a quota between cells, keep
    the total at 40 and still be authoritative. Now frozen:
    `AllocationProfile.splits` is a schema-derived property (not a stored field,
    so it cannot disagree with its own schema); the late-band cell/geometry
    comes from the profile; schema-5 `phase_allocation` must equal
    `allocation_for_schema(5)` **per cell**, with a correct grand total
    explicitly insufficient; missing/extra cells, altered counts, non-`"all"`
    splits and unknown roles are refused; and the schema-5 fingerprint records
    the effective assignment strategy so an artifact cannot be reinterpreted
    under the other path.
55. **The portable fixture was not constructible.** The v17 config binds
    `source_index_path`, `protocol_path`, `replay_dir`, `match_summary_path` and
    `screen_out` plus an `expected_fingerprints` block, so pairing it verbatim
    with a reduced screen makes the authenticated chain refuse — and the listed
    fixture lacked the source index and sidecars anyway. Contract (a) is chosen:
    a **selector-core** fixture using a tracked fixture profile and reduced rows,
    exercising `post_screen_qualification_report` and `sample_v2_rows` only, and
    explicitly not called a full producer regression. Coverage corrected from
    "all four roles' v2 equivalents" (v17 has **two** roles) to both roles, both
    splits, all four phases and every late band.
56. **Task 6 now operationalizes the bundle.** Six runnable builder tests
    (canonical/atomic/reproducible emission, computed `byte_identical`, exact key
    set, returned-not-embedded digest) and an explicit Step 3 requirement.
    Tamper attacks stay with `load_verified_a6400_bundle`, since a builder cannot
    be attacked by editing its own output.
57. Revision 8 item 47 marked superseded on the self-digest point.

## Revision 11 change log

Revision 10 was reviewed and rejected for execution. Two P1s and one P2, all
incorporated:

58. **The late-band profile source was undefined.** Revision 10 said floors come
    from "the profile, never `LATE_TARGET_CELL`", but `band_minima_total` and
    `band_minima_per_split` (`:273-274`) hold only counts — the cell they
    constrain comes from `LATE_TARGET_CELL` (`:616`, consumed at `:404`, `:432`,
    `:703`, `:1140`, `:1258`). Added `AllocationProfile.band_floor_cell`, a
    schema-derived property returning `("target", "late")` for schemas 1–4 and
    `None` for schema 5, with non-empty minima under a `None` cell a parse
    error. All floor accounting and reporting route through it; the module
    constant keeps its value and assertions.
59. **The bundle's own digest vanished before the verdict.** `evaluate` now takes
    the bundle **path**, hashes the bytes once, records
    `a6400_reference_bundle_sha1`, verifies **those exact bytes** without a
    re-read, and takes both capture hashes from the verified result — so Stage 0
    binds the accepted document rather than reconstructing it.
60. **The advertised tests were not executable or independent.** `RUN1`/`RUN2`/
    `RUN2_DIFFERENT` now come from a `captures` fixture writing three small
    capture-schema documents; the key-set test compares against an independent
    literal (`EXPECTED_BUNDLE_KEYS`) with a separate test pinning the
    implementation tuple to it, so an added key plus an edited tuple cannot
    pass; the atomicity test forces the failure at `os.replace` rather than on a
    nonexistent input — revision 10's version failed before any write was
    attempted and proved nothing — and a second test proves an existing
    destination is left unchanged; Task 9's five tamper attacks are named
    individually; and every schema-5 parser refusal gets a parameterized test.

## Revision 12 change log

Revision 11 was reviewed and rejected for execution. Two P1s and one P2, all
incorporated:

61. **The builder's valid-input contract contradicted its own tests.**
    `authenticate_against` requires the exact 30-case set and raises otherwise,
    so a one-case fixture could never satisfy the positive builder tests; and a
    bundle with one `authentication` block is undefined as to which run it
    describes when the captures differ. The builder now **accepts only
    byte-identical captures that both authenticate against all 30 cases**, and
    raises without writing otherwise; `byte_identical` becomes an invariant of a
    valid bundle rather than a variable, still independently recomputed by the
    verifier. A failed duplicate run may be recorded under a separate,
    explicitly non-binding artifact kind. The builder is split into a pure
    `build_a6400_bundle_document` plus a production wrapper, and the fixtures
    now supply 30 cases with a monkeypatched frozen source loader.
62. **The verifier API was path-based and byte-based at once.** Frozen as
    `verify_a6400_bundle_bytes(raw, *, bundle_path)` — the real verifier, with
    `bundle_path` retained for resolving and restricting capture paths — plus an
    optional `load_verified_a6400_bundle(path)` that reads once and delegates.
    `evaluate` performs the single read, hashes `raw`, and passes that same
    bytes object to the byte verifier.
63. **Schema-5 refusal tests are now runnable entries** in
    `tests/test_fpu_dev_corpus_v2_v18_hook.py` — a parameterized case per
    rejection reason, plus positive tests for exact allocation authority, the
    shifted-quota-same-total case, `splits`/`band_floor_cell` under schema 5 and
    under legacy schemas, the fingerprint strategy field, and one-split
    assignment. The stale claim that the five tamper attacks also run on the
    producer side is removed; they live only with the verifier.

## Revision 13 change log

Revision 12 was reviewed: **scientifically approved, not yet executable.** Two
P1s and one P2, all test-fixture and API completion:

64. **The bundle's authentication block had no defined provenance.**
    `authenticate_against` was exact-or-raise returning nothing, while the
    bundle had to store 30 entries. Added
    `authentication_report(source_rows, captured) -> list[dict]` (pure, total,
    case_id-ordered) and made `authenticate_against` **return** it after its
    exact-or-raise validation. The builder authenticates both captures and,
    because their bytes are identical, requires the two reports to be
    canonically identical before storing one — a divergence would mean
    authentication is not a pure function of the capture bytes.
65. **The 30-case fixture was not portable.** It called
    `M.testing.synthetic_capture`, a test-only namespace that does not exist and
    should not be added to a production module, and the real identities live
    under gitignored `logs/`. Replaced with a tracked fixture at
    `tests/golden/a6400_bundle_fixture/{source_rows.json,capture.json}` loaded
    by local helpers in the test module, plus a separate test that
    `_load_frozen_a6400_source` enforces the production path and the
    `a17d4737…` hash — since every other test monkeypatches it.
66. **The schema-5 block was not runnable.** `BASE5 = {...}` was a set
    containing `Ellipsis`; it is now a `base5()` factory returning a fresh valid
    dict, mutated through `copy.deepcopy` so nested edits cannot leak between
    parameterized cases. The duplicated `target|opening` mutation is resolved —
    "extra cell" is dropped and "out-of-table role/phase" keeps that input, so
    each case has a distinct trigger. `sample_v2_rows` is unpacked as
    `(rows, stats)` per its `Tuple[List[dict], dict]` signature at `:1392-1394`.
    `legacy_profile_for(schema)` was proposed here and is **withdrawn in
    revision 14** — see item 68.

## Revision 14 change log

Revision 13 was reviewed: scientifically approved, four executability blockers.
All incorporated:

67. **`base5()` was rejected by the parser before reaching any schema-5 logic.**
    `required_keys` includes `late_floors` and `corpus_size`, and schema ≥3
    demands `scientific_interpretation_forbidden` as a bool equal to
    `interpretation_forbidden(run_kind)` (`:349-367`). The factory now supplies
    all of them, and the **run kind and label are frozen together**: schema-5
    run kind is `"v18_preflight_sizing"`, its label is `true`, and
    `"production"` is neither permitted nor parseable. Added a guard test that
    `base5()` is actually accepted, plus refusals for non-v18 run kinds and for
    an interpretable label.
68. **`legacy_profile_for` is withdrawn, not added.** The module constants
    canonically describe only the **schema-1** legacy path; schemas 2–4 are
    config-authoritative, and the live v17 schema-4 config has a five-cell
    `phase_allocation` rather than `SPLIT_ALLOC_V2`'s eight. Synthesizing one
    would let a green fingerprint coexist with drifted real behaviour. Schema 1
    uses the existing `AllocationProfile.legacy()`; schemas 2–4 are tested from
    **tracked real profiles** captured in Step 0.
69. **The frozen-source path test is behavioural.** It instruments the real
    read and hash boundary and asserts the exact path used, with a positive
    exact-path case, a hash-mismatch refusal, and a signature test proving the
    loader takes no path argument — so substitution is impossible by
    construction rather than by convention. Revision 13 asserted only that the
    path appeared in a docstring, which a byte-identical copy elsewhere would
    also satisfy.
71. *(Revision 15 supersedes item 70's "populated for every schema" clause and
    item 67's delegation to `interpretation_forbidden`.)*

70. **The one-split test checks the assignment map.** It asserts
    `stats["split_assignment"]` covers **every retained game** and maps each to
    `"all"`. Checking only the 40 selected rows would pass even if a two-way
    assigner placed half the reservoir in `frozen_check`.

## Global Constraints

Every task's requirements implicitly include this section.

- **Branch:** `v18-depth2-provisional-backup`, branched from `fpu-v17-baseline-policy-mass`. Not from `main` — the v17 machinery this plan imports exists only on the v17 branch.
- **NO `mcts.py` EDIT.** Read-only throughout. **The check is against the pinned v18 baseline, never against `main`** — this branch inherits v17's MCTS changes, so `git diff main..HEAD -- scripts/GPU/alphazero/mcts.py` is 62 lines (52 insertions, 10 deletions) and an "empty diff vs main" assertion can never pass. Task 0 pins the branch point and the file's content SHA-1; every later check uses those.
- **NO positive-cap search.** No task runs MCTS with a v18 cap active; the field does not exist yet. Every search here is shipped search.
- **NO commit without explicit per-task authorization from the user.** Each task ends with "request authorization, then commit" — stop and ask.
- **NO measurement until every task is committed at a clean HEAD.** See the Execution Phase.
- **Run Python as** `.venv/bin/python`; tests as `.venv/bin/python -m pytest -p no:cacheprovider`.
- **Never call `sys.modules.pop("mlx")` in a test** — a later fresh `import mlx.core` re-initialises the native Metal module and deterministically SIGABRTs the suite.
- **`logs/` is gitignored** (`.gitignore:31`). Every artifact is local-only; bind by recorded SHA-1.
- **Every emitted artifact is canonical and timestamp-free** via `fpu_dev_reservoir_protocol.canonical_json_bytes`, and **every preflight artifact stamps `scientific_interpretation_forbidden: true` explicitly** — never relying on prose.
- **Frozen batching triple** `(eval_batch_size, stall_flush_sims, pending_virtual_visits) = (14, 48, 8)`, asserted and recorded although inert on the synchronous path.
- **Frozen cap grid:** identity `2.00`; candidates `1.25, 1.00, 0.75, 0.50`. Strongest `0.50`, weakest candidate `1.25`.
- **Historical anchors — exact values only, never rounded forms:**
  - v16a held-out `-0.20`: `0.28027286567513765` (`1 - 24583/34156`), artifact `6d15c7dd15bdc8e8a983700f536950bcc9830019`
  - v17 weakest `r=0.15`: `0.23358985966500678` (`516/2209`), artifact `af7778c84e1ea04f463febfc615e5363400d6aad`
  - v16 selected-A FPU `-0.20`: `0.81836179163573375` (`3307/4041`), artifact `f201f0f25b868e5c4c7103992054c7b4df5074d1`
- **A rows establish reach only.** Every numeric threshold derives from the non-A control cohort (spec §2.2.3). Task 9 has a test that must be able to fail on this.
- **Game identity is the replay's content SHA-1**, via `diagnose_fpu_baseline_policy_mass.game_identities` (`:1694`). `game_idx` is reservoir-local: comparing raw indices both invents overlaps and misses a copied game that was renumbered. Never identify a game by `(replay_dir, game_idx)`.
- **At most one control position per game.** This removes within-game correlation at the source, so no game-clustered bootstrap is required. With an 800-game universe the cohort is not supply-constrained.

## Implementation order

**Execute in this order, which is NOT the numeric order:**

```text
0 -> 1 -> 2 -> 3 -> 5 -> 4 -> 4b -> 6 -> 7 -> 8 -> 9
```

Task 5 creates `v18_preflight_criteria`, which Tasks 4 and 4b **import**: Task 4
needs `UNIVERSE` and `CENSUS`, Task 4b needs `MATCHING`. Revision 5 placed Task 5
after both, so those imports could not exist and the constants would have been
restated — the exact duplication the design forbids. Each importing task carries
an object-identity test proving it holds the criteria module's object rather than
a copy.

Task numbers are kept stable so earlier review references still resolve.

---

### Task 0: Pin the v18 baseline

**Files:** Create `logs/eval/v18_depth2_provisional_backup/v18_baseline_pin.json` (gitignored; bound by SHA-1 in every later artifact).

Nothing may be built until the "no `mcts.py` edit" invariant is expressible. Record, before any other work:

- [ ] **Step 1: Capture the branch point and file identity**

```bash
git rev-parse HEAD                                   # v18 branch point
shasum -a 1 scripts/GPU/alphazero/mcts.py            # content identity
git rev-list --count main..HEAD                      # inherited v17 commits
```

Expected at the time of writing: branch point `345ec93`, `mcts.py` content SHA-1
`b60c983399dbc5ed292de9b15944b8850a1d8508`, 34 commits ahead of `main`. Verify
rather than assume — if HEAD has moved, the pinned values change and this file
records the new ones.

- [ ] **Step 2: Write the pin and define the invariant**

Write the three values plus `run_kind: "v18_baseline_pin"` and
`scientific_interpretation_forbidden: true` canonically. The invariant used by
every later task is:

```text
shasum -a 1 scripts/GPU/alphazero/mcts.py == <pinned content SHA-1>
git diff --stat <pinned branch point>..HEAD -- scripts/GPU/alphazero/mcts.py  is empty
```

- [ ] **Step 3: Request authorization, then commit the pin reference**

The artifact itself is gitignored; commit only any code or docs referencing it.

## File Structure

| Path | Responsibility |
|---|---|
| `scripts/GPU/alphazero/v18_provisional_backup.py` | **Create.** The single clip-formula implementation. Pure. Later imported by `mcts.py`. |
| `scripts/GPU/alphazero/v18_tree_walk.py` | **Create.** Read-only metric derivation from a finished tree. |
| `scripts/GPU/alphazero/v18_crossover.py` | **Create.** Static first-order crossover, signed, with synchronous-provenance assertion. |
| `scripts/GPU/alphazero/v18_control_pool.py` | **Create.** Establishes, verifies and freezes the exact-800-game non-A **source universe** and its per-game census plan. Selects no cohort. |
| `scripts/GPU/alphazero/v18_cohort_matcher.py` | **Create.** Post-measurement 1:1 minimum-cost cohort matcher over game-columns; emits the authenticated matched-cohort artifact. |
| `scripts/GPU/alphazero/v18_preflight_criteria.py` | **Create.** Every frozen threshold, formula and decision rule; canonical emission. |
| `scripts/GPU/alphazero/diagnose_v18_residual_preflight.py` | **Create.** Measurement only. Emits artifacts, computes no verdict. |
| `scripts/GPU/alphazero/v18_selector_sizing.py` | **Create.** Residual screen telemetry, selector predicates, sizing ladder, operating characteristics. |
| `scripts/GPU/alphazero/v18_preflight_verdict.py` | **Create.** Pure evaluator: artifacts + criteria → verdict and derived thresholds. |
| `scripts/GPU/alphazero/capture_v17_abcd_selected_moves.py` | **Modify.** Parameterize sims, gate subset and authentication source. |
| `tests/test_v18_*.py`, `tests/test_capture_abcd_parameterization.py` | **Create.** One test module per task. |

---

### Task 1: The single clip-formula implementation

**Files:**
- Create: `scripts/GPU/alphazero/v18_provisional_backup.py`
- Test: `tests/test_v18_provisional_backup.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `IDENTITY_CAP: float = 2.0`; `CAP_GRID: tuple[float, ...] = (1.25, 1.00, 0.75, 0.50)`; frozen dataclass `ProvisionalBackup(backup_value: float, residual: float, clipped_amount: float, clip_direction: int)`; `provisional_depth2_backup_value(raw_leaf_value: float, raw_parent_value: float, cap: float) -> ProvisionalBackup`.

**Perspective convention — pinned here because revision 1 got it wrong.** Spec §1.1 reports the top positive depth-1 child at raw **black** value about `-0.087`. That node is **red to move**, so its parent-to-move value — the value actually stored in `parent.nn_value` and passed as `raw_parent_value` — is `+0.087`. The leaf below it is black to move at `+0.793`. Therefore:

```text
raw_parent_value = +0.087          (parent-to-move perspective)
baseline         = -raw_parent     = -0.087   (leaf perspective)
residual         = 0.793 - (-0.087) = 0.880
backup at cap 0.50 = -0.087 + 0.50  = 0.413
```

Every task in this plan uses `+0.087 / 0.880 / 0.413`. A test asserting `0.706` has the parent sign reversed.

- [ ] **Step 1: Write the failing test**

Create `tests/test_v18_provisional_backup.py`:

```python
"""v18 clip formula -- spec Sec 3.1 / 4.2. Pure; no MCTS, no GPU."""
import pytest

from scripts.GPU.alphazero.v18_provisional_backup import (
    CAP_GRID,
    IDENTITY_CAP,
    ProvisionalBackup,
    provisional_depth2_backup_value,
)

# The observed A pattern, spec Sec 1.1. The depth-1 node is RED to move, so its
# stored parent-to-move value is +0.087 even though its black value is -0.087.
A_PARENT_TO_MOVE = 0.087
A_LEAF_TO_MOVE = 0.793
A_RESIDUAL = 0.880
A_BACKUP_AT_050 = 0.413


def test_no_clip_returns_the_original_leaf_object_value():
    out = provisional_depth2_backup_value(0.20, A_PARENT_TO_MOVE, 1.25)
    assert out.backup_value == 0.20          # exact, not baseline+residual
    assert out.clipped_amount == 0.0
    assert out.clip_direction == 0


def test_no_clip_does_not_reconstruct_arithmetically():
    leaf, parent = 0.1 + 0.2, 0.3
    out = provisional_depth2_backup_value(leaf, parent, 2.0 - 1e-9)
    assert repr(out.backup_value) == repr(leaf)


def test_positive_clip_matches_the_observed_a_pattern():
    out = provisional_depth2_backup_value(A_LEAF_TO_MOVE, A_PARENT_TO_MOVE, 0.50)
    assert out.residual == pytest.approx(A_RESIDUAL, abs=1e-12)
    assert out.clip_direction == 1
    assert out.backup_value == pytest.approx(A_BACKUP_AT_050, abs=1e-12)
    assert out.clipped_amount == pytest.approx(A_RESIDUAL - 0.50, abs=1e-12)


def test_the_observed_a_residual_does_not_bind_at_the_weakest_cap():
    # 0.880 < 1.25. This is why the ladder expects the weakest cap to under-reach
    # rather than to be unsafe -- spec Sec 7 routes that to "advance", not "reject".
    out = provisional_depth2_backup_value(A_LEAF_TO_MOVE, A_PARENT_TO_MOVE, 1.25)
    assert out.clip_direction == 0


def test_negative_clip_is_symmetric_in_magnitude():
    out = provisional_depth2_backup_value(-A_LEAF_TO_MOVE, -A_PARENT_TO_MOVE, 0.50)
    assert out.residual == pytest.approx(-A_RESIDUAL, abs=1e-12)
    assert out.clip_direction == -1
    assert out.backup_value == pytest.approx(-A_BACKUP_AT_050, abs=1e-12)
    assert out.clipped_amount == pytest.approx(A_RESIDUAL - 0.50, abs=1e-12)


def test_boundary_is_strict_so_abs_residual_equal_cap_does_not_clip():
    # Spec Sec 9.2.1 identity witnesses (max|residual| <= 0.50 vs strongest cap
    # 0.50) depend on this boundary being exact.
    out = provisional_depth2_backup_value(0.50, 0.0, 0.50)
    assert out.residual == 0.50
    assert out.clip_direction == 0
    assert out.backup_value == 0.50


def test_perspective_reversal_symmetry_over_the_whole_grid():
    for cap in CAP_GRID:
        for leaf, parent in [(0.9, -0.1), (-0.4, 0.7), (0.05, 0.05), (1.0, -1.0)]:
            a = provisional_depth2_backup_value(leaf, parent, cap)
            b = provisional_depth2_backup_value(-leaf, -parent, cap)
            assert b.backup_value == pytest.approx(-a.backup_value, abs=1e-15)
            assert b.residual == pytest.approx(-a.residual, abs=1e-15)
            assert b.clipped_amount == pytest.approx(a.clipped_amount, abs=1e-15)
            assert b.clip_direction == -a.clip_direction


def test_identity_cap_cannot_bind_on_any_tanh_pair():
    for leaf in (-1.0, -0.3, 0.0, 0.3, 1.0):
        for parent in (-1.0, -0.3, 0.0, 0.3, 1.0):
            out = provisional_depth2_backup_value(leaf, parent, IDENTITY_CAP)
            assert out.clip_direction == 0
            assert out.backup_value == leaf


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_inputs_raise_rather_than_bypass(bad):
    with pytest.raises(ValueError):
        provisional_depth2_backup_value(bad, 0.0, 1.0)
    with pytest.raises(ValueError):
        provisional_depth2_backup_value(0.0, bad, 1.0)


@pytest.mark.parametrize("bad", [0.0, -0.5, 2.0001, float("nan")])
def test_out_of_range_cap_rejected(bad):
    with pytest.raises(ValueError):
        provisional_depth2_backup_value(0.5, 0.0, bad)


@pytest.mark.parametrize("bad", [True, False])
def test_boolean_cap_rejected(bad):
    with pytest.raises(ValueError):
        provisional_depth2_backup_value(0.5, 0.0, bad)


def test_result_is_immutable():
    out = provisional_depth2_backup_value(0.9, -0.1, 0.5)
    assert isinstance(out, ProvisionalBackup)
    with pytest.raises(Exception):
        out.backup_value = 0.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest -p no:cacheprovider tests/test_v18_provisional_backup.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.GPU.alphazero.v18_provisional_backup'`.

- [ ] **Step 3: Write the implementation**

Create `scripts/GPU/alphazero/v18_provisional_backup.py`:

```python
"""v18 depth-2 provisional backup -- the SINGLE implementation of the formula.

Spec: docs/superpowers/specs/2026-07-29-v18-depth2-provisional-backup-design.md
Sec 3.1 (formula), Sec 3.2 (eligibility), Sec 4.2 (helper contract).

Pure: no node traversal, no evaluator access, no RNG, no state mutation. Safe to
import from the preflight, the screen, the diagnostic and (later) from `mcts.py`.
Spec Sec 4.2 requires exactly one implementation: import this, never copy it.

PERSPECTIVE CONTRACT -- stated once, unmistakably:

    raw_leaf_value   is in the LEAF's to-move perspective
    raw_parent_value is in the PARENT's to-move perspective (the opposite side)

Worked example from the measured A pattern (spec Sec 1.1). The top positive
depth-1 child has raw BLACK value about -0.087, but it is RED to move, so
`parent.nn_value` holds +0.087. The leaf below is black to move at +0.793:

    baseline = -(+0.087) = -0.087
    residual = 0.793 - (-0.087) = 0.880
    backup at cap 0.50 = -0.087 + 0.50 = 0.413

A depth-2 leaf shares the root's side to move, so the returned backup value is in
the same perspective as `raw_leaf_value` and drops straight into the existing
sign-alternating `MCTS._backup` path.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# Explicit identity sentinel. Spec Sec 4.1: two tanh values differ by at most
# 2.0, so this cap can never bind -- but the real identity guarantee is the
# structural branch at the call site, which returns before reaching this module.
IDENTITY_CAP: float = 2.0

# Spec Sec 7, weakest candidate first, matching the weakest-first ladder.
CAP_GRID: tuple[float, ...] = (1.25, 1.00, 0.75, 0.50)


@dataclass(frozen=True)
class ProvisionalBackup:
    """One clip decision.

    `backup_value` enters `_backup`. The other three fields exist so the call
    site fills Sec 4.4's telemetry columns WITHOUT recomputing the residual -- a
    second computation is a second implementation.

    `clip_direction` is +1 when an unusually POSITIVE first estimate was pulled
    down, -1 when an unusually NEGATIVE one was pulled up, 0 when the cap did not
    bind. `clipped_amount` is always non-negative.
    """

    backup_value: float
    residual: float
    clipped_amount: float
    clip_direction: int


def _check_cap(cap: float) -> None:
    # bool subclasses int, so `True` would silently act as cap 1.0.
    if isinstance(cap, bool):
        raise ValueError(f"cap must be a float, not a bool: {cap!r}")
    if not isinstance(cap, (int, float)) or not math.isfinite(cap):
        raise ValueError(f"cap must be finite: {cap!r}")
    if cap <= 0.0 or cap > IDENTITY_CAP:
        raise ValueError(f"cap must satisfy 0.0 < cap <= {IDENTITY_CAP}; got {cap!r}")


def provisional_depth2_backup_value(
    raw_leaf_value: float,
    raw_parent_value: float,
    cap: float,
) -> ProvisionalBackup:
    """Clip a depth-2 leaf's raw value toward its parent's raw value.

    Spec Sec 3.1. The comparison is STRICT: `abs(residual) == cap` does not clip.
    """
    _check_cap(cap)
    if isinstance(raw_leaf_value, bool) or isinstance(raw_parent_value, bool):
        raise ValueError("raw values must be floats, not bools")
    if not math.isfinite(raw_leaf_value) or not math.isfinite(raw_parent_value):
        # Spec Sec 3.2: fail loudly. Bypassing would conceal corruption.
        raise ValueError(
            f"nonfinite raw value: leaf={raw_leaf_value!r} parent={raw_parent_value!r}")

    baseline = -raw_parent_value          # parent value in the LEAF's perspective
    residual = raw_leaf_value - baseline

    if residual > cap:
        return ProvisionalBackup(baseline + cap, residual, residual - cap, 1)
    if residual < -cap:
        return ProvisionalBackup(baseline - cap, residual, -cap - residual, -1)

    # Spec Sec 3.1: return the ORIGINAL value, never `baseline + residual`, so an
    # unbound row carries no rounding difference from shipped.
    return ProvisionalBackup(raw_leaf_value, residual, 0.0, 0)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest -p no:cacheprovider tests/test_v18_provisional_backup.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest -p no:cacheprovider -q`
Record the pre-task baseline count first; expected: baseline + new tests, zero failures.

- [ ] **Step 6: Request authorization, then commit**

```bash
git add scripts/GPU/alphazero/v18_provisional_backup.py tests/test_v18_provisional_backup.py
git commit -m "feat(v18): single implementation of the depth-2 provisional backup formula"
```

---

### Task 2: Read-only tree walker

**Files:**
- Create: `scripts/GPU/alphazero/v18_tree_walk.py`
- Test: `tests/test_v18_tree_walk.py`

**Interfaces:**
- Consumes: `v18_provisional_backup.{provisional_depth2_backup_value, IDENTITY_CAP}`; `mcts.{MCTSNode, visit_leader_move}` read-only.
- Produces: `terminating_backups(node) -> int`; `eligible_depth2_pairs(root) -> list[tuple[MCTSNode, MCTSNode]]`; `residual(parent, leaf) -> float`; `would_clip(root, cap) -> list[MCTSNode]`; `depth_terminating_histogram(root) -> dict[int, int]`; `leader(root) -> MCTSNode | None`; `replies(leader_node) -> int`; `explored_replies(leader_node) -> list[MCTSNode]`; `follow_up_visits_per_explored_reply(leader_node) -> float`; `revisit_to_depth3_rate(root, cap) -> float`; `positive_mass(root) -> float`; `negative_mass(root) -> float`; `sign_dominance(root) -> float`; `contribution_weighted_positive_mass(root, cap) -> float`; `exposed_positive_backup_mass(root, cap) -> tuple[float, float]` returning `(numerator, denominator)`; `terminal_depth2_counts(root) -> tuple[int, int]`; `walk(root, caps) -> dict`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_v18_tree_walk.py`:

```python
"""v18 read-only tree walker -- spec Sec 4.4 and Sec 10.1.1.

Constructed trees only: no evaluator, no GPU. `MCTSNode` needs a `state`
supporting `.is_terminal()`, which is all these tests exercise.
"""
import pytest

from scripts.GPU.alphazero.mcts import MCTSNode
from scripts.GPU.alphazero import v18_tree_walk as W


class FakeState:
    def __init__(self, terminal=False):
        self._terminal = terminal

    def is_terminal(self):
        return self._terminal


def node(nn_value=None, visits=0, value_sum=0.0, terminal=False, priors=None):
    n = MCTSNode(state=FakeState(terminal))
    n.nn_value = nn_value
    n.visit_count = visits
    n.value_sum = value_sum
    n.priors = {} if (priors is None and terminal) else (
        {0: 1.0} if priors is None else priors)
    return n


def attach(parent, child, move_id):
    parent.children[move_id] = child
    child.parent = parent
    child.move = move_id
    return child


def build_tree():
    """root -> two depth-1 children; the first carries three depth-2 leaves.

    Visit counts chosen so the terminating identity is hand-checkable:
      root 10; d1a 7, d1b 3; under d1a: leaf_hi 3, leaf_lo 2, leaf_term 1
      terminating(root)    = 10 - (7 + 3)     = 0
      terminating(d1a)     = 7  - (3 + 2 + 1) = 1
      terminating(leaf_hi) = 3  - 0           = 3

    d1a.nn_value = +0.087 (parent-to-move), so the baseline for its leaves is
    -0.087 and leaf_hi's residual is 0.793 - (-0.087) = 0.880 -- the measured A
    pattern from spec Sec 1.1.
    """
    root = node(nn_value=0.10, visits=10, value_sum=1.0)
    d1a = attach(root, node(nn_value=0.087, visits=7, value_sum=-1.0), 1)
    d1b = attach(root, node(nn_value=0.05, visits=3, value_sum=-0.2), 2)
    leaf_hi = attach(d1a, node(nn_value=0.793, visits=3, value_sum=2.379), 11)
    leaf_lo = attach(d1a, node(nn_value=0.10, visits=2, value_sum=0.20), 12)
    leaf_term = attach(d1a, node(visits=1, value_sum=1.0, terminal=True), 13)
    return root, d1a, d1b, leaf_hi, leaf_lo, leaf_term


def test_terminating_backups_identity():
    root, d1a, _d1b, leaf_hi, _lo, _t = build_tree()
    assert W.terminating_backups(root) == 0
    assert W.terminating_backups(d1a) == 1
    assert W.terminating_backups(leaf_hi) == 3


def test_depth_histogram_sums_to_root_visit_count():
    root, *_ = build_tree()
    assert sum(W.depth_terminating_histogram(root).values()) == root.visit_count


def test_eligible_pairs_exclude_terminal_leaves():
    root, d1a, _d1b, leaf_hi, leaf_lo, leaf_term = build_tree()
    leaves = [leaf for _p, leaf in W.eligible_depth2_pairs(root)]
    assert leaf_hi in leaves and leaf_lo in leaves and leaf_term not in leaves


def test_eligible_pairs_exclude_empty_prior_leaves():
    # Spec Sec 3.2: _expand_batch assigns a synthetic nn_value 0.0 to a node with
    # no legal moves; such a leaf is bypassed, not clipped.
    root, d1a, *_ = build_tree()
    synthetic = attach(d1a, node(nn_value=0.0, visits=1, priors={}), 14)
    assert synthetic not in [leaf for _p, leaf in W.eligible_depth2_pairs(root)]


def test_empty_parent_priors_raises_rather_than_skips():
    root = node(nn_value=0.0, visits=3)
    bad_parent = attach(root, node(nn_value=0.1, visits=2, priors={}), 1)
    attach(bad_parent, node(nn_value=0.5, visits=1), 11)
    with pytest.raises(ValueError):
        W.eligible_depth2_pairs(root)


def test_residual_uses_the_negated_parent_baseline():
    _root, d1a, _d1b, leaf_hi, *_ = build_tree()
    assert W.residual(d1a, leaf_hi) == pytest.approx(0.880, abs=1e-12)


def test_would_clip_is_cap_monotone():
    root, *_ = build_tree()
    # residuals present: leaf_hi 0.880, leaf_lo 0.187
    assert len(W.would_clip(root, 1.25)) == 0
    assert len(W.would_clip(root, 0.75)) == 1
    assert len(W.would_clip(root, 0.50)) == 1
    assert len(W.would_clip(root, 0.10)) == 2


def test_would_clip_is_defined_on_a_tree_that_never_clipped():
    """The point of the would_clip population: a shipped tree has zero ACTUAL
    clips, yet the counterfactual population is well defined and non-empty."""
    root, *_ = build_tree()
    assert W.would_clip(root, 0.50)


def test_leader_and_replies_include_terminals():
    root, d1a, *_ = build_tree()
    assert W.leader(root) is d1a
    # replies is the imported v17 breadth statistic: terminals INCLUDED.
    assert W.replies(d1a) == 3


def test_explored_replies_exclude_terminal_and_ineligible():
    # `MCTSNode` is a plain @dataclass, so eq=True sets __hash__ = None and the
    # type is UNHASHABLE: a set of nodes raises TypeError while building the
    # expected value, before the walker's return value is ever examined.
    # Compare identities instead -- that imposes neither hashability nor
    # ordering on the declared `list[MCTSNode]` return type.
    _root, d1a, _d1b, leaf_hi, leaf_lo, _t = build_tree()
    got = W.explored_replies(d1a)
    assert {id(n) for n in got} == {id(leaf_hi), id(leaf_lo)}


def test_follow_up_visits_counts_visits_after_first_touch():
    _root, d1a, *_ = build_tree()
    # leaf_hi 3 visits -> 2 follow-ups; leaf_lo 2 -> 1. Mean over 2 replies = 1.5
    assert W.follow_up_visits_per_explored_reply(d1a) == pytest.approx(1.5)


def test_follow_up_visits_empty_denominator_is_invalid_not_zero():
    with pytest.raises(ValueError):
        W.follow_up_visits_per_explored_reply(node(nn_value=0.1, visits=1))


def test_revisit_rate_over_would_clip_population():
    root, _d1a, _d1b, leaf_hi, *_ = build_tree()
    attach(leaf_hi, node(nn_value=-0.2, visits=1), 111)
    assert W.revisit_to_depth3_rate(root, 0.10) == pytest.approx(0.5)


def test_revisit_rate_empty_denominator_is_invalid_not_zero():
    root, *_ = build_tree()
    with pytest.raises(ValueError):
        W.revisit_to_depth3_rate(root, 1.25)


def test_sign_dominance_formula_and_zero_denominator():
    root, *_ = build_tree()
    pos, neg = W.positive_mass(root), W.negative_mass(root)
    assert pos > 0 and neg == 0
    assert W.sign_dominance(root) == pytest.approx(pos / (pos + neg))
    # A tree with no residual mass at all scores 0.0, never a ZeroDivisionError.
    flat_root = node(nn_value=0.0, visits=2)
    flat_d1 = attach(flat_root, node(nn_value=0.0, visits=1), 1)
    attach(flat_d1, node(nn_value=0.0, visits=1), 11)
    assert W.sign_dominance(flat_root) == 0.0


def test_exposed_positive_backup_mass_numerator_and_denominator():
    root, *_ = build_tree()
    num, den = W.exposed_positive_backup_mass(root, 0.50)
    # denominator sums terminating_backups * max(0, raw_leaf) over ALL eligible
    # leaves: leaf_hi 3*0.793 + leaf_lo 2*0.10 = 2.579
    assert den == pytest.approx(3 * 0.793 + 2 * 0.10, abs=1e-12)
    # numerator restricts to leaves that would clip at 0.50: leaf_hi only.
    assert num == pytest.approx(3 * 0.793, abs=1e-12)


def test_terminal_depth2_counts():
    root, *_ = build_tree()
    terminal, total = W.terminal_depth2_counts(root)
    assert terminal == 1
    assert total == 3          # leaf_hi, leaf_lo, leaf_term (all visited)


def test_walk_emits_every_documented_key_per_cap():
    root, *_ = build_tree()
    rec = W.walk(root, caps=(1.25, 0.50))
    for key in ("root_visit_count", "depth_terminating_histogram",
                "depth_ge3_backups", "depth_ge3_fraction", "leader_move",
                "replies", "explored_replies", "follow_up_visits_per_reply",
                "eligible_depth2_leaves", "positive_mass", "negative_mass",
                "sign_dominance", "terminal_depth2", "total_depth2", "per_cap"):
        assert key in rec, key
    assert set(rec["per_cap"]) == {"1.25", "0.5"}
    for cap_rec in rec["per_cap"].values():
        for key in ("would_clip_count", "clipped_amount_total", "positive_count",
                    "negative_count", "revisit_to_depth3_rate",
                    "contribution_weighted_positive_mass",
                    "exposed_positive_mass_numerator",
                    "exposed_positive_mass_denominator"):
            assert key in cap_rec, key
    # Cap 1.25 reaches no leaf on this tree, so its revisit rate has an empty
    # denominator. `walk` records null rather than fabricating 0.0 or aborting
    # the whole multi-cap record -- 0.0 would read as "clipped leaves were never
    # revisited", the opposite of "nothing was clipped".
    assert rec["per_cap"]["1.25"]["would_clip_count"] == 0
    assert rec["per_cap"]["1.25"]["revisit_to_depth3_rate"] is None


def test_walk_is_deterministic_and_json_safe():
    import json
    root, *_ = build_tree()
    a, b = W.walk(root, caps=(0.50,)), W.walk(root, caps=(0.50,))
    assert a == b
    json.dumps(a)          # no sets, no node objects
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest -p no:cacheprovider tests/test_v18_tree_walk.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the implementation**

Create `scripts/GPU/alphazero/v18_tree_walk.py` implementing exactly the interface above. Binding details, all pinned by the tests:

- `terminating_backups(n) = n.visit_count - sum(c.visit_count for c in n.children.values())`. Exact on the synchronous path, which adds no virtual visits to `visit_count`; the batched path is out of v18 scope (spec §4.3).
- `depth_terminating_histogram` walks breadth-first from the root; must sum to `root.visit_count`.
- `depth_ge3_backups` sums depth `>= 3`; `depth_ge3_fraction` divides by `root.visit_count` (spec §10.1.1: denominator is the full budget, identical across arms).
- Eligibility (spec §3.2): path exactly `[root, parent, leaf]`; leaf nonterminal; `leaf.priors` non-empty; `leaf.nn_value` and `parent.nn_value` finite. **Empty `parent.priors` raises `ValueError`** — corruption, not a skip.
- `residual(parent, leaf)` returns `provisional_depth2_backup_value(leaf.nn_value, parent.nn_value, IDENTITY_CAP).residual`, so the residual comes from the single formula implementation.
- `would_clip(root, cap)` returns eligible leaves with `clip_direction != 0`.
- `replies` counts children with `visit_count > 0`, terminals **included**; `explored_replies` excludes terminals and empty-prior nodes.
- `follow_up_visits_per_explored_reply` = `sum(c.visit_count - 1) / len(explored_replies)`; raise on an empty denominator.
- `revisit_to_depth3_rate(root, cap)` = fraction of `would_clip(root, cap)` with at least one child having `visit_count > 0`; raise on an empty denominator.
- `positive_mass(root) = sum(max(0, residual_i))`, `negative_mass(root) = sum(max(0, -residual_i))` over eligible leaves. `sign_dominance = positive_mass / (positive_mass + negative_mass)`, returning `0.0` when the denominator is zero.
- `contribution_weighted_positive_mass(root, cap) = sum over eligible leaves of (terminating_backups(leaf) / root.visit_count) * max(0, residual_i - cap)`.
- `exposed_positive_backup_mass(root, cap)` returns `(numerator, denominator)` where the denominator sums `terminating_backups(leaf) * max(0, leaf.nn_value)` over **all** eligible leaves and the numerator restricts to leaves with `residual > cap`. Returning both, rather than the ratio, lets the caller pool across rows instead of averaging ratios.
- `terminal_depth2_counts(root)` returns `(terminal_visited, all_visited)` over depth-2 children of every depth-1 node.
- `walk(root, caps)` returns a JSON-serialisable, deterministic dict with the keys the test enumerates; `per_cap` keyed by `str(cap)`.
- **Undefined statistics are recorded as `None` (JSON `null`), never as a fabricated `0.0`.** The standalone functions keep raising `ValueError` on an empty denominator — that contract is unchanged, and callers computing a single statistic still get the loud failure. `walk` is the one caller that must not abort, because a weak cap reaching no leaf is an expected outcome of a multi-cap walk and must not destroy the records of the caps beside it. Concretely: `revisit_to_depth3_rate` is `null` whenever `would_clip_count == 0`, and `follow_up_visits_per_reply` is `null` when the leader has no explored replies (or there is no leader). The distinction is load-bearing for the ladder — `0.0` would read as "the clipped leaves were never revisited", the opposite of "no leaf was clipped".

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest -p no:cacheprovider tests/test_v18_tree_walk.py -v`
Expected: PASS.

- [ ] **Step 5: Prove the walker agrees with a real search tree**

Read `tests/fpu_search_fixture.py` and reuse its fake evaluator — do not build a new one. Add to the same test module an integration test that runs **shipped** search via `MCTS(evaluator, cfg, random.Random(7)).search_with_root(state, add_noise=False)` and asserts:

- `sum(W.depth_terminating_histogram(root).values()) == root.visit_count == cfg.n_simulations`
- `W.would_clip(root, 2.0) == []`
- `W.walk(root, caps=(0.50,))` is JSON-serialisable

Run: `.venv/bin/python -m pytest -p no:cacheprovider tests/test_v18_tree_walk.py -v`
Expected: PASS. This is what stops the hand-built trees from lying about real tree shape.

- [ ] **Step 6: Run the full suite, then request authorization and commit**

```bash
git add scripts/GPU/alphazero/v18_tree_walk.py tests/test_v18_tree_walk.py
git commit -m "feat(v18): read-only tree walker deriving all v18 metrics post hoc"
```

---

### Task 3: Static first-order crossover analysis (signed)

**Files:**
- Create: `scripts/GPU/alphazero/v18_crossover.py`
- Test: `tests/test_v18_crossover.py`

**Interfaces:**
- Consumes: `v18_provisional_backup.provisional_depth2_backup_value`; `v18_tree_walk`; `mcts.MCTSNode`.
- Produces: `assert_synchronous_tree(root, expected_sims, *, search_execution_mode) -> None`; `counterfactual_child_q(parent, child, cap) -> float`; `crossover_at_node(parent, cap, c_puct) -> dict`; `crossover_for_tree(root, cap, c_puct) -> dict` returning `{"predicted_shipped_replies": int, "predicted_capped_replies": int, "predicted_reply_delta": int, "predicted_reply_reduction": float, "per_node": [...]}`.

**The delta is signed.** The clip is symmetric, so a large *negative* residual lowers a visited child's counterfactual score and can *increase* reply scanning. Therefore:

```text
predicted_reply_reduction =
    (predicted_shipped_replies - predicted_capped_replies) / predicted_shipped_replies
```

may be negative, and a negative value is a **meaningful scientific result**, not invalid input. Nothing in this module or downstream may clamp it, and `R_min` derivation must not convert a negative prediction into a positive floor (Task 5).

- [ ] **Step 1: Write the failing test**

Create `tests/test_v18_crossover.py`:

```python
"""v18 static first-order crossover analysis -- spec Sec 2.2.1."""
import pytest

from scripts.GPU.alphazero import v18_crossover as X
from tests.test_v18_tree_walk import attach, build_tree, node


def test_counterfactual_substitutes_the_initial_contribution_only():
    """Spec Sec 2.2.1: value_sum - nn_value + clipped_initial.

    leaf_hi: nn_value 0.793, visits 3, value_sum 2.379.
    Parent nn_value +0.087 -> baseline -0.087, residual 0.880.
    At cap 0.50 the clipped initial value is -0.087 + 0.50 = 0.413.
    Counterfactual sum = 2.379 - 0.793 + 0.413 = 1.999; q = 1.999 / 3.
    """
    _root, d1a, _d1b, leaf_hi, *_ = build_tree()
    assert X.counterfactual_child_q(d1a, leaf_hi, 0.50) == pytest.approx(
        1.999 / 3, abs=1e-12)


def test_counterfactual_equals_actual_q_when_the_cap_does_not_bind():
    _root, d1a, _d1b, leaf_hi, *_ = build_tree()
    assert X.counterfactual_child_q(d1a, leaf_hi, 2.0) == pytest.approx(
        leaf_hi.value_sum / leaf_hi.visit_count, abs=1e-15)


def test_synchronous_assertion_rejects_a_tree_with_the_wrong_visit_count():
    root, *_ = build_tree()          # root.visit_count == 10
    X.assert_synchronous_tree(root, 10, search_execution_mode="synchronous")
    with pytest.raises(ValueError):
        X.assert_synchronous_tree(root, 400, search_execution_mode="synchronous")


def test_synchronous_assertion_refuses_a_batched_tree_with_a_MATCHING_count():
    """The load-bearing case: identical visit count, wrong provenance.

    `search_from_root` backs up EVERY waiter on a pending leaf with the same
    expansion value (mcts.py:595-606), so a leaf's `value_sum` can hold
    k*nn_value while `counterfactual_child_q` substitutes exactly one -- the
    substitution is then wrong by (k-1)*(backup_value - nn_value). But the
    batched path still backs up one path per simulation, so
    `root.visit_count == expected_sims` holds on BOTH paths. The count proves
    the simulation BUDGET and never the provenance; only the mode does.
    """
    root, *_ = build_tree()
    with pytest.raises(ValueError):
        X.assert_synchronous_tree(root, 10, search_execution_mode="batched_waiter")


def test_synchronous_assertion_has_no_default_mode_and_rejects_unknown_modes():
    # A default of "synchronous" would reinstate exactly the hole this closes:
    # every caller that forgot the argument would silently assert the safe
    # value. The argument is required and keyword-only.
    root, *_ = build_tree()
    with pytest.raises(TypeError):
        X.assert_synchronous_tree(root, 10)
    for bad in ("Synchronous", "sync", "", None, True):
        with pytest.raises(ValueError):
            X.assert_synchronous_tree(root, 10, search_execution_mode=bad)


def test_identity_cap_predicts_no_change():
    root, *_ = build_tree()
    out = X.crossover_for_tree(root, cap=2.0, c_puct=1.5)
    assert out["predicted_reply_delta"] == 0
    assert out["predicted_reply_reduction"] == 0.0


def test_reduction_may_be_negative_for_a_negative_residual_population():
    """A large NEGATIVE residual lowers the counterfactual visited score, which
    makes unvisited replies relatively MORE attractive -- more scanning, not
    less. The plan must not clamp this away.

    The unvisited priors are load-bearing and must not be "tidied". The count
    can only move for an unvisited move whose score falls between the capped and
    the shipped best visited score:

        band            = (1.004309, 1.076532]
        unvisited score = c_puct * prior * sqrt(19 + 1) = 6.708204 * prior
        => only prior in (0.149714, 0.160480] can move the count

    So 13's prior is 0.155, and 12 absorbs the remainder to keep the priors
    summing to 1.0. With the obvious-looking {12: 0.3, 13: 0.2} BOTH unvisited
    moves already outscore the shipped best (2.012461 and 1.341641), the counts
    are 2 and 2, and the assertion `0.0 < 0.0` is unreachable by construction --
    the mechanism is real but the fixture cannot express it.
    """
    root = node(nn_value=0.0, visits=20, value_sum=0.0)
    d1 = attach(root, node(nn_value=-0.9, visits=19, value_sum=1.0,
                           priors={11: 0.5, 12: 0.345, 13: 0.155}), 1)
    # leaf raw -0.9 against baseline +0.9 -> residual -1.8, binds hard at 0.50.
    attach(d1, node(nn_value=-0.9, visits=18, value_sum=-16.2), 11)
    out = X.crossover_for_tree(root, cap=0.50, c_puct=1.5)
    assert out["predicted_reply_reduction"] < 0.0
    # Pin the exact outcome: a sign flip or any clamp fails HERE, rather than
    # silently degrading to the vacuous 0.0 == 0.0 the original fixture gave.
    assert out["predicted_shipped_replies"] == 1
    assert out["predicted_capped_replies"] == 2
    assert out["predicted_reply_delta"] == -1
    assert out["predicted_reply_reduction"] == -1.0


def test_reduction_is_the_documented_ratio():
    root, *_ = build_tree()
    out = X.crossover_for_tree(root, cap=0.50, c_puct=1.5)
    s, c = out["predicted_shipped_replies"], out["predicted_capped_replies"]
    assert out["predicted_reply_delta"] == s - c
    assert out["predicted_reply_reduction"] == pytest.approx((s - c) / s)


def test_zero_shipped_denominator_is_invalid_not_zero():
    lone = node(nn_value=0.0, visits=1)
    with pytest.raises(ValueError):
        X.crossover_for_tree(lone, cap=0.50, c_puct=1.5)


def test_crossover_excludes_terminal_and_synthetic_children():
    root, d1a, *_ = build_tree()
    attach(d1a, node(nn_value=0.0, visits=1, priors={}), 14)
    out = X.crossover_at_node(d1a, cap=0.50, c_puct=1.5)
    assert out["excluded_terminal"] >= 1
    assert out["excluded_synthetic"] >= 1


def test_crossover_is_deterministic():
    root, *_ = build_tree()
    assert (X.crossover_for_tree(root, cap=0.75, c_puct=1.5)
            == X.crossover_for_tree(root, cap=0.75, c_puct=1.5))
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest -p no:cacheprovider tests/test_v18_crossover.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the implementation**

Create `scripts/GPU/alphazero/v18_crossover.py`. Core:

```python
def counterfactual_child_q(parent, child, cap):
    """Child Q under the counterfactual that its FIRST backup had been clipped.

    A visited child's q_value is a running mean, not its raw evaluation, so the
    clipped value must be substituted into the accumulated sum. EXACT because the
    expansion backup contributed precisely `child.nn_value` to `child.value_sum`
    (mcts.py:1145-1148) -- see `assert_synchronous_tree` for the prerequisite that
    makes that true. A batched-waiter tree can back one expansion to several
    waiters, and the substitution would NOT be exact there.
    """
    pb = provisional_depth2_backup_value(child.nn_value, parent.nn_value, cap)
    return (child.value_sum - child.nn_value + pb.backup_value) / child.visit_count
```

`crossover_at_node(parent, cap, c_puct)` mirrors `_select_child` (`mcts.py:1091-1114`):

```text
sqrt_parent          = sqrt(parent.visit_count + 1)
visited_score(child) = -q(child) + c_puct * prior[child] * sqrt_parent / (1 + child.visit_count)
                       with q = actual for shipped, counterfactual for capped
unvisited_score(m)   = 0.0 + c_puct * prior[m] * sqrt_parent
```

It returns, for shipped and for the cap, the best visited score and the count of unvisited priors whose `unvisited_score` exceeds it — these are `predicted_shipped_replies` / `predicted_capped_replies` at that node — plus `excluded_terminal` and `excluded_synthetic`. `crossover_for_tree` sums over eligible depth-1 nodes and computes the signed ratio above; it raises `ValueError` on a zero shipped denominator rather than returning zero.

`assert_synchronous_tree(root, expected_sims, *, search_execution_mode)` raises `ValueError` unless **both** hold:

1. `root.visit_count == expected_sims`, and
2. `search_execution_mode == "synchronous"` exactly.

`search_execution_mode` is **required and keyword-only, with no default**. A default of `"synchronous"` would reinstate the hole this closes — every caller that omitted the argument would silently assert the safe value — so omitting it must raise `TypeError`, and any other value (`"batched_waiter"`, `None`, `True`, `"sync"`, a case variant) must raise `ValueError`.

**Why the count alone is not evidence.** Both search entry points back up exactly one path per simulation, so `root.visit_count == expected_sims` holds identically on either — the count proves the simulation *budget*, never the provenance. The distinction that matters is elsewhere: `search_from_root` "backs up ALL waiters with the returned value" when a pending leaf is expanded (`mcts.py:595-606`), so a leaf with `k` waiters accumulates `k * nn_value` into `value_sum`, while `counterfactual_child_q` subtracts exactly one `nn_value`. The substitution is then wrong by `(k-1) * (backup_value - nn_value)` and the whole crossover analysis silently loses exactness. `search_with_root` is documented as "the same synchronous per-sim path as `search()`; NOT `search_from_root`'s batched waiter path" (`mcts.py:528-535`), which is the only path on which the substitution is exact.

The mode is therefore an **input to be proven by the caller's route** (Task 7), not a property this function can read off the tree. Nothing in this module may infer it, default it, or derive it from node state. The prerequisite is documented verbatim from spec §2.2.1. **The preflight asserts this rather than assuming it.**

The module docstring must state in bold what the analysis **cannot** do (spec §2.2.1): it does not reproduce sequential selection once the candidate tree diverges, so it cannot empirically derive conversion efficiency.

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest -p no:cacheprovider tests/test_v18_crossover.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full suite, then request authorization and commit**

```bash
git add scripts/GPU/alphazero/v18_crossover.py tests/test_v18_crossover.py
git commit -m "feat(v18): signed static first-order crossover with synchronous-provenance assertion"
```

---

### Task 4: Freeze the authenticated broad source universe

**Revision 3 restructure.** Revision 2 had this task select the final control
cohort on CPU using "the same near-even and minimum-eligible-leaf shape the
future selector will use". That is impossible from replay data. A replay move
carries only `col, n_legal, player, ply, root_top1_share, root_total_visits,
root_value, row, selected_visit_count, selected_visit_rank` — no tree, no
depth-2 population, no residuals — and the stored `root_value` belongs to
whichever checkpoint held that colour (the file records `black_checkpoint` and
`red_checkpoint` separately), so in a `0379_vs_calib020_0001` pool it is
`0379`'s value on one colour. A near-even rule cannot be applied here at all.

The corrected flow, with the matcher's variables and tolerances frozen in Task 5
**before** any search runs:

```text
Task 4  freeze a broad SOURCE UNIVERSE using immutable replay predicates only
        (ply window, side, n_legal, termination reason, exclusions)
Task 7  run shipped calib020_0001 search over a per-game CENSUS of that universe
Task 4b precommitted matcher picks the final non-A cohort from REMEASURED fields
        -- phase, side, near-even root value, branching, eligible-leaf count --
        and NEVER from residual magnitude
```

The same measured census also feeds Task 8's sizing, so one measurement pass
serves both purposes.



Revision 1 named `logs/eval/calib020_post_opening_sweep/position_probe_cases.csv` as the control source. **Verified and rejected:** it holds 240 rows over only **30 distinct positions** (`case_id` count 30, `(game_idx, position_ply)` count 30) repeated across checkpoints `0001 0003 0005 0008 0010 0015 0379 0409`, and those 30 positions *are* gate C. Canonically excluding established C positions leaves zero controls; keeping only `checkpoint=0001` retains duplicate evaluations of the same consumed positions rather than independent discovery controls.

This task builds a real pool and proves its exclusions rather than asserting them.

**Files:**
- Create: `scripts/GPU/alphazero/v18_control_pool.py`
- Test: `tests/test_v18_control_pool.py`

**Interfaces:**
- Consumes: `fpu_state_hash.canonical_state_sha1`, `position_probe_cases.position_state`, `fpu_provenance.{file_sha1, replay_data_sha1}`, `canonical_json_bytes`.
- Produces: `FORBIDDEN_SOURCES: tuple[dict, ...]`; `CANDIDATE_UNIVERSES: tuple[dict, ...]`; `forbidden_canonical_hashes() -> set[str]`; `forbidden_game_content_sha1s() -> set[str]`; `enumerate_census(universe_spec) -> list[dict]`; `apply_exclusions(rows, forbidden_hashes, forbidden_games) -> tuple[list[dict], dict]`; `freeze_source_universe(out_path, universe_name, seed) -> dict`.

**Exclusion sets, computed not assumed.** Two exclusions are applied:

- **Canonical position**, via `fpu_state_hash.canonical_state_sha1`.
- **Game**, via the replay's **content SHA-1** using the established
  `diagnose_fpu_baseline_policy_mass.game_identities` (`:1694`). Revision 2 used
  `(replay_dir, game_idx)`, which regresses a lesson v17 already learned and
  recorded in that helper's own docstring: `game_idx` is reservoir-local, so raw
  indices "both invent overlaps and miss a copied game that was renumbered."
  Import the helper; do not re-derive identity.

**Source binding.** Binding the directory is insufficient. The frozen record
binds the match summary, the 800-row JSONL, every replay sidecar, both
checkpoint identities, and the replay-content hashes, cross-checked against each
other.

**The selected source has NO fixed colour assignment.** `seed20116` is
colour-balanced: each checkpoint plays black in 400 games and red in the other
400. A `black_checkpoint_sha1` / `red_checkpoint_sha1` pair would therefore be a
false claim, and checking only the first filename-sorted replay would conclude a
fixed assignment and be wrong about 400 games. The record binds instead:

```text
checkpoint_sha1s     {path: sha1} for the unordered PAIR
anchor_checkpoint    calib020_0001, the anchor both colours are measured under
games_by_colour      black: {anchor: 400, opponent: 400}
                     red:   {anchor: 400, opponent: 400}
```

Both colours are counted for both roles, and BOTH `a_as_black.games` and
`a_as_red.games` from the summary are cross-checked -- a one-sided count leaves
the other colour unverified, which is where an imbalance would hide.

**AUTHENTICATED BYTES MUST BE THE BYTES PARSED.** Hashing a path and then
reopening it is not authentication: a file that changes between the two reads
supplies census rows derived from bytes the record does not describe, and the
artifact then carries a hash for content it never measured. Every pinned
artifact is read ONCE into a buffer, hashed from that buffer, and parsed from
that same buffer; the exclusion sets and the census consume those buffers and
never reopen a path. Where an established helper must read from disk -- the
length-delimited `fpu_provenance.replay_data_sha1` over 800 sidecars -- the
contract is completed by a CLOSING re-authentication after the census has
consumed the snapshot and BEFORE any bytes are written, so drift is refused with
no partial artifact left behind.

**The forbidden REPLAY RESERVOIRS are bound too.** Authenticating the probe CSVs
and manifests is not sufficient: the canonical exclusion hashes are
*reconstructed from* the sidecars those files point at, so a replay that drifts
changes the exclusion SET while every evidence-file hash in the record stays
unchanged. Three reservoirs are referenced, each pinned by game count and by the
established length-delimited aggregate:

```text
seed20115   gate_A, gate_D, v16a_neutral, and the A game identities
            427d4ab669a81fe409de7da6d7c458056aff306e   800 games
seed35791   gate_B
            d36b01c0993095e07785666316028f0c875eed7b   800 games
seed40937   gate_C
            80aa2068319cdbe0429100b736d293f5b8bc437e   800 games
```

Every canonical exclusion and every A game identity is derived INSIDE an
opening/closing window over all three, and a referenced replay lying outside
every pinned reservoir is a hard refusal -- a gate source that starts pointing
somewhere new fails loudly rather than contributing silently unauthenticated
exclusions.

Forbidden sources — every consumed or acceptance population:

| source | why |
|---|---|
| A probe cases (`calib020_0001_black_loss_post_opening_predrop_probe`) | the reach population; must not also supply thresholds |
| gate B (`black_predrop_calib010_goal_line`), C (`calib020_post_opening_sweep`), D (`calib020_0001_red_loss_post_opening_predrop_probe`) | established acceptance positions (spec §2.2) |
| v16a neutral manifest (`logs/eval/v16a_fpu_unbiased/neutral_position_manifest.csv`) | consumed; do-not-repeat #42 forbids tuning against it |
| v16 production corpus + v17 development corpus (selected rows) | consumed selection records (spec §2.1) |
| replay dir `calib020_0001_vs_0379_800g_w4_seed20115_replays` | the A rows' game source — game-level exclusion |

Candidate pools, in preference order, each to be **verified** at run time rather than trusted:

1. `logs/eval/0379_vs_calib020_0001_800g_w4_seed20116_replays` — 800 games under the same `calib020_0001` anchor, seed range disjoint from A's `seed20115`, from v16 production (`20300000+`) and from v17 development (`20310000+`). Retired at the *corpus-composition* stage for branching-band geometry, never consumed as evidence.
2. The authenticated v17 development reservoir (`logs/eval/fpu_v17_baseline_policy_mass/development`), **minus** its 32 selected rows.
3. The v16 production reservoir (`logs/eval/fpu_v16_policy_mass_v2/production_v2_b400amend_4000g_seed20300000`), **minus** its 120 selected rows.

- [ ] **Step 1: Write the failing test**

Create `tests/test_v18_control_pool.py`:

```python
"""Non-A discovery control pool -- spec Sec 2.2 and Sec 2.2.3.

The pool supplies EVERY numeric threshold, so its independence from A and from
established acceptance positions is the load-bearing property here.
"""
import json
import pytest

from scripts.GPU.alphazero import v18_control_pool as P


def test_forbidden_sources_cover_a_and_all_four_gates():
    names = {s["name"] for s in P.FORBIDDEN_SOURCES}
    for required in ("gate_A", "gate_B", "gate_C", "gate_D",
                     "v16a_neutral_consumed", "v16_production_selected",
                     "v17_development_selected", "a_replay_games"):
        assert required in names, required


def test_gate_c_source_is_recorded_as_degenerate_and_forbidden():
    """Revision 1 proposed this as the CONTROL source. It is 240 rows over 30
    distinct positions that are exactly gate C. Pin the finding so it cannot be
    silently reintroduced."""
    c = next(s for s in P.FORBIDDEN_SOURCES if s["name"] == "gate_C")
    assert c["distinct_positions"] == 30
    assert c["total_rows"] == 240
    assert c["rejected_as_control_source"] is True


def test_apply_exclusions_removes_canonical_hash_matches():
    rows = [{"canonical_sha1": "a" * 40, "game_content_sha1": "1" * 40},
            {"canonical_sha1": "b" * 40, "game_content_sha1": "2" * 40}]
    kept, report = P.apply_exclusions(rows, {"a" * 40}, set())
    assert [r["canonical_sha1"] for r in kept] == ["b" * 40]
    assert report["excluded_by_canonical_hash"] == 1


def test_apply_exclusions_removes_whole_games_by_replay_content_sha1():
    """Game identity is the replay's CONTENT hash, never (dir, game_idx).
    `game_idx` is reservoir-local, so index comparison both invents overlaps and
    misses a copied game that was renumbered -- the lesson already recorded in
    diagnose_fpu_baseline_policy_mass.game_identities:1694."""
    rows = [{"canonical_sha1": "c" * 40, "game_content_sha1": "7" * 40},
            {"canonical_sha1": "e" * 40, "game_content_sha1": "8" * 40}]
    kept, report = P.apply_exclusions(rows, set(), {"7" * 40})
    assert [r["game_content_sha1"] for r in kept] == ["8" * 40]
    assert report["excluded_by_game"] == 1


def test_game_identity_helper_is_imported_not_reimplemented():
    from scripts.GPU.alphazero import diagnose_fpu_baseline_policy_mass as D
    assert P.game_identities is D.game_identities


def test_renumbered_copy_of_a_forbidden_game_is_still_excluded():
    rows = [{"canonical_sha1": "a" * 40, "game_content_sha1": "7" * 40,
             "game_idx": 999, "replay_dir": "some/other/dir"}]
    kept, _report = P.apply_exclusions(rows, set(), {"7" * 40})
    assert kept == []


def test_exclusion_report_is_non_vacuous():
    """A pool whose exclusions remove nothing has not been verified. The freeze
    must record counts so a zero is visible rather than implied."""
    rows = [{"canonical_sha1": "f" * 40, "game_content_sha1": "1" * 40}]
    _kept, report = P.apply_exclusions(rows, set(), set())
    assert report["excluded_by_canonical_hash"] == 0
    assert report["excluded_by_game"] == 0
    assert report["input_rows"] == 1


def test_freeze_refuses_a_universe_that_collapses_to_empty():
    with pytest.raises(ValueError, match="collapsed"):
        P.freeze_source_universe(out_path="/tmp/x.json",
                                 universe_name="__empty__", seed=20260729)


def test_freeze_refuses_when_distinct_games_are_fewer_than_the_minimum():
    """The failure mode that killed the revision-1 control source: plenty of
    ROWS, far too few distinct games. Because the cohort takes at most one
    position per game, distinct GAMES is the binding supply."""
    with pytest.raises(ValueError, match="distinct"):
        P.freeze_source_universe(out_path="/tmp/x.json",
                                 universe_name="__degenerate__", seed=20260729)


def test_frozen_universe_record_is_byte_reproducible(tmp_path):
    a = P.freeze_source_universe(str(tmp_path / "a.json"), "__fixture__", 20260729)
    b = P.freeze_source_universe(str(tmp_path / "b.json"), "__fixture__", 20260729)
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()
    assert a["universe_sha1"] == b["universe_sha1"]


def test_frozen_universe_record_stamps_the_scope_labels(tmp_path):
    rec = P.freeze_source_universe(str(tmp_path / "a.json"), "__fixture__", 20260729)
    assert rec["run_kind"] == "shipped_only_preflight_source_universe"
    assert rec["scientific_interpretation_forbidden"] is True
    assert rec["selection_is_independent_of_residual_exposure"] is True


def test_universe_is_exactly_800_games(): ...
def test_fewer_than_800_eligible_games_refuses(): ...
def test_zero_yield_games_are_retained_in_all_game_ids(): ...
def test_position_exclusions_never_drop_a_game_from_all_game_ids(): ...
def test_census_phase_allocation_is_1_1_1_3(): ...
def test_missing_phase_contributes_zero_and_is_reported(): ...
def test_missing_phase_is_never_backfilled_from_another_phase(): ...
def test_single_slot_phases_use_q_one_half_ceil_index(): ...
def test_late_slots_use_q_quarter_half_three_quarter(): ...
def test_census_ceiling_of_4800_aborts_before_evaluator_load(): ...
def test_criteria_constants_are_imported_not_restated(): ...
    # object identity: P.UNIVERSE is C.UNIVERSE and P.CENSUS is C.CENSUS


def test_universe_binds_summary_jsonl_sidecars_and_checkpoints(tmp_path):
    """Binding the directory is insufficient -- bind the artifacts."""
    rec = P.freeze_source_universe(str(tmp_path / "a.json"), "__fixture__", 20260729)
    for key in ("summary_sha1", "jsonl_sha1", "replay_data_sha1",
                "checkpoint_sha1s", "anchor_checkpoint", "games_by_colour"):
        assert key in rec, key
    # The source alternates colours 400/400, so a fixed black/red identity is a
    # false claim and must be ABSENT, not merely unused.
    assert "black_checkpoint_sha1" not in rec
    assert "red_checkpoint_sha1" not in rec


# Authentication boundary -- each must fail if its guard is removed.
def test_jsonl_and_sidecar_disagreement_is_refused(): ...
def test_checkpoint_pair_mutation_is_refused(): ...
def test_colour_schedule_imbalance_is_refused(): ...
def test_mutation_between_authentication_and_census_is_detected(): ...
    # end to end on the real universe: opening auth passes, the census consumes
    # the snapshot, the sidecars drift, the CLOSING re-authentication refuses,
    # and no artifact is written
def test_no_partial_artifact_when_the_freeze_refuses(): ...
def test_exclusions_refuse_unauthenticated_payloads(): ...
def test_authenticated_bytes_are_the_bytes_parsed(): ...
    # one read: the buffer that was hashed is the buffer that gets parsed
def test_forbidden_replay_reservoirs_are_bound(): ...
def test_every_referenced_replay_lies_in_a_pinned_reservoir(): ...
def test_gate_b_replay_drift_leaves_no_artifact(): ...
def test_gate_c_replay_drift_leaves_no_artifact(): ...
def test_a_game_identity_source_drift_leaves_no_artifact(): ...
def test_closing_check_covers_every_replay_source(): ...
def test_jsonl_hash_gate_still_refuses_an_unpinned_mutation(): ...
def test_report_refuses_any_real_universe_other_than_the_selected_one(): ...
def test_report_authenticates_and_uses_the_snapshot(): ...
    # spies assert forbidden payloads authenticated once, reservoirs bracketed
    # ["opening", "closing"], and the 800-replay snapshot reaching enumeration
def test_report_refuses_on_selected_universe_drift(): ...
def test_report_refuses_on_forbidden_reservoir_drift(): ...
    # `test_jsonl_and_sidecar_disagreement_is_refused` REPINS the expected JSONL
    # hash to the mutated bytes, so authentication passes and the cross-field
    # comparison is what refuses. Without the repin the hash gate fires first
    # and the test would stay green even with the comparison deleted.


def test_freeze_emits_a_census_but_selects_no_cohort(tmp_path):
    rec = P.freeze_source_universe(str(tmp_path / "a.json"), "__fixture__", 20260729)
    assert "census_positions" in rec
    assert "matched_cohort" not in rec        # Task 4b's job, after measurement


def test_selection_never_reads_residual_exposure(tmp_path):
    """The universe must be frozen WITHOUT looking at the statistic it will later
    calibrate, or the threshold is fitted to its own sample. Nothing here could
    read residuals anyway -- they do not exist pre-search -- so this test also
    documents that invariant."""
    rec = P.freeze_source_universe(str(tmp_path / "a.json"), "__fixture__", 20260729)
    assert "exposure" not in json.dumps(rec["selection_inputs"]).lower()
    assert "residual" not in json.dumps(rec["selection_inputs"]).lower()
```

The `__fixture__`, `__empty__` and `__degenerate__` names are test-only entries in `CANDIDATE_UNIVERSES`, backed by small inline fixtures so these tests need no replay data and no GPU. The `__degenerate__` fixture reproduces the revision-1 failure shape: many rows, few distinct games.

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest -p no:cacheprovider tests/test_v18_control_pool.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the implementation**

Create `scripts/GPU/alphazero/v18_control_pool.py`. Requirements:

- `FORBIDDEN_SOURCES` records each source's path, its `distinct_positions` and `total_rows` **as measured**, and for gate C additionally `rejected_as_control_source: True` with the reason.
- `forbidden_canonical_hashes()` reconstructs every forbidden position via `position_state` and hashes it with `canonical_state_sha1`.
- `forbidden_game_content_sha1s()` returns replay-content SHA-1s, computed with the imported `game_identities` helper. Re-export it as `P.game_identities` so the test can assert it is the same object, not a copy.
- `enumerate_census` walks the universe's replays and proposes a **per-game census** using **only** fields present in the replay: `ply` window, `player` (side), `n_legal` (branching), the game's `reason`/`winner`, and the per-game move count. It applies **no** value-based rule — `root_value` is checkpoint-contaminated — and **no** residual rule, which does not exist pre-search. Record the exact criteria in `selection_inputs`.
- `freeze_source_universe` applies both exclusions, then refuses with `ValueError("... collapsed ...")` on an empty result and `ValueError("... distinct ...")` when distinct **games** after exclusion are fewer than the frozen minimum. Because the cohort takes at most one position per game, distinct games — not rows — is the binding supply. On success it writes a canonical record with the exclusion report, the universe SHA-1, the replay-data SHA-1, every source SHA-1 listed above, git commit, worktree state, `run_kind`, and `scientific_interpretation_forbidden: True`.
- **`freeze_source_universe` selects no cohort.** It freezes what may be measured. The cohort is chosen by Task 4b after measurement.

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest -p no:cacheprovider tests/test_v18_control_pool.py -v`
Expected: PASS.

- [ ] **Step 5: STOP — the user chooses the universe source and configuration**

**The report is read-only, not unauthenticated.** It runs the SAME
opening/closing authentication as the freeze — forbidden payloads, all three
forbidden reservoirs, and the selected universe's snapshot handed to
`enumerate_census` rather than letting it reread the sidecars — and writes
nothing. Figures derived from unverified bytes are a guess, not a report.

**Once the source is chosen, real reports accept only that source.** Before the
decision the report enumerates candidate 1, then 2, then 3 as needed; after
`SELECTED_UNIVERSE` is bound in tracked code, enumerating an alternative real
source is a post-selection inspection route and is refused. Fixtures stay
available to the unit tests.

Run the enumeration and exclusion **report only** against candidate universe 1 (no search, no GPU — replay parsing and hashing):

```bash
.venv/bin/python -m scripts.GPU.alphazero.v18_control_pool --report --universe seed20116
```

Present: distinct **games** surviving both exclusions (the binding supply, since the cohort takes at most one position per game), distinct positions, side balance, per-phase census yield, and both exclusion counts. **The user chooses the source and configuration here.** If universe 1 yields fewer than 800 eligible games, report universe 2, then 3.

**This step chooses; it does not freeze.** The binding artifact cannot be written yet, because Task 4's own module and tests are still uncommitted and the record would capture a dirty tree and a stale HEAD — the same defect revision 2 had with the criteria artifact. The frozen universe record is emitted at **Execution Phase step 3**, at the committed clean HEAD, from the source the user selects now.

- [ ] **Step 6: Request authorization, then commit**

```bash
git add scripts/GPU/alphazero/v18_control_pool.py tests/test_v18_control_pool.py
git commit -m "feat(v18): authenticated broad source universe with canonical and content-SHA game exclusions"
```

---

### Task 4b: Precommitted cohort matcher

Selects the final non-A control cohort **after** Task 7's measurement, using only
fields the frozen matcher is permitted to see. Its rules are frozen in Task 5
before any search runs, so this module contains logic but no thresholds.

**Files:**
- Create: `scripts/GPU/alphazero/v18_cohort_matcher.py`
- Test: `tests/test_v18_cohort_matcher.py`

**Interfaces:**
- Consumes: `v18_preflight_criteria.MATCHING`, the frozen universe record, the Task 7 `census_positions.csv`.
- Produces: `hungarian_rectangular(cost) -> list[tuple[int, int]]` (vendored; **no scipy in this venv**); `match_cohort(census_rows, a_rows, matching) -> tuple[list[dict], dict]`; `emit_matched_cohort(out_path, ...) -> str` returning the artifact SHA-1.

**Exact cardinality, frozen before measurement:** 1:1, `n_A = n_C = 30`. This is what makes Task 5's AUC operating-characteristic table computable and approvable *now*, before any GPU run. A complete matching of all 30 or a `PREFLIGHT_FAIL` — never a cohort of 29.

**Deterministic minimum-cost bipartite matching**, not greedy. Greedy nearest-neighbour can fail on A-row ordering when a complete valid matching exists, converting a solvable problem into a spurious failure. Inadmissible pairs (outside any tolerance) carry infinite cost and can never be matched; cost is the sum of per-variable absolute differences each divided by its own tolerance; ties break lexicographically on `(cost, control canonical_state_sha1, control game_content_sha1, control position_ply)`.

**Task 4b emits an authenticated artifact.** Revision 3 returned rows in memory, which Tasks 8 and 9 could not bind. `emit_matched_cohort` canonically writes the exact matched rows, the complete matching report (per-variable balance, per-pair cost, unmatched A rows), and the SHA-1s of the universe record, `census_positions.csv`, the frozen criteria and the A source, plus the matching algorithm name and version, the cardinality, `run_kind` and `scientific_interpretation_forbidden: true`. Tasks 8 and 9 authenticate this artifact by SHA-1 before reading it.

Permitted matching variables: `phase`, `side`, `|root_value_stm|`, `n_legal`, `eligible_depth2_leaves`. **Forbidden:** residual magnitude, exposure, `would_clip` counts, clipped amount — anything the cohort will later calibrate.

- [ ] **Step 1: Write the failing test**

Create `tests/test_v18_cohort_matcher.py`, each written out in full:

```python
def test_matcher_reads_no_residual_field(): ...          # inspect the row keys it touches
def test_matcher_enforces_at_most_one_position_per_game(): ...
def test_game_identity_is_the_replay_content_sha1_not_game_idx(): ...
def test_matcher_is_deterministic_and_order_independent(): ...
def test_matcher_respects_every_frozen_tolerance(): ...
def test_matcher_finds_a_complete_matching_that_greedy_would_miss(): ...
def test_matcher_refuses_unless_all_30_are_matched(): ...
def test_matcher_reports_unmatched_a_rows_rather_than_dropping_them(): ...
def test_matching_report_records_per_variable_balance_and_per_pair_cost(): ...
def test_two_a_rows_preferring_different_positions_from_one_game(): ...
def test_global_tie_break_is_pinned_under_multiple_optima(): ...
def test_vendored_hungarian_agrees_with_brute_force_on_small_matrices(): ...
def test_hungarian_rectangular_matches_a_known_optimum(): ...
def test_infinite_cost_pairs_can_never_be_matched(): ...
def test_emitted_cohort_artifact_is_byte_reproducible(): ...
def test_emitted_artifact_binds_universe_census_criteria_and_a_source_sha1s(): ...
def test_emitted_artifact_records_algorithm_version_cardinality_and_run_kind(): ...
```

Four are load-bearing:

- `test_matcher_reads_no_residual_field` builds census rows whose residual fields carry values that *would* change the outcome if consulted, and asserts the cohort is unchanged. It must fail if the matcher reads them.
- `test_matcher_finds_a_complete_matching_that_greedy_would_miss` constructs the case where the first A row's nearest control is the only admissible control for a later A row: greedy returns 29 and fails, min-cost returns 30.
- `test_two_a_rows_preferring_different_positions_from_one_game` is the one that catches revision 4's bug: two A rows whose best admissible positions are two *different* plies of the **same** game. A position-column matrix assigns both and silently violates one-per-game; a game-column matrix cannot.
- `test_vendored_hungarian_agrees_with_brute_force_on_small_matrices` enumerates all permutations for `n <= 6`, including matrices with `inf` cells and with multiple optima, and asserts the vendored solver's total cost and its tie-broken assignment match. This is the correctness evidence for hand-rolling an algorithm rather than importing one.

- [ ] **Step 2: Run the test to verify it fails.** Expected: module not found.

- [ ] **Step 3: Write the implementation.** Vendor a rectangular Hungarian algorithm (~40 lines; O(n²m) on a 30 × 800 matrix is trivial) — `scipy` is **not** installed in this venv, so `linear_sum_assignment` is unavailable and adding scipy for one call is not justified.

**Columns are GAMES, not positions.** Build a `30 × distinct_games` cost matrix. Cell `(a, g)` holds game `g`'s **best admissible position** for A row `a` and that position's cost, or `math.inf` if game `g` offers no admissible position for `a`. Hungarian then assigns each game at most once **structurally**, which is what enforces one control per game.

Revision 4's approach — reduce to the lowest-cost position per game *per A row*, then solve a position-column matrix — does **not** enforce it: two A rows can still be assigned two different positions drawn from the same game, because those are distinct columns. The invariant has to live in the matrix shape, not in a pre-filter.

Solve, then refuse with `ValueError` unless all 30 A rows are matched at finite cost. Emit via `emit_matched_cohort`.

- [ ] **Step 4: Run the test, then the full suite.** Expected: PASS, zero failures.

- [ ] **Step 5: Request authorization, then commit**

```bash
git add scripts/GPU/alphazero/v18_cohort_matcher.py tests/test_v18_cohort_matcher.py
git commit -m "feat(v18): precommitted post-measurement cohort matcher"
```

---

### Task 5: Frozen preflight criteria — the preregistration

> **Execute this task BEFORE Tasks 4 and 4b.** They import `UNIVERSE`, `CENSUS`
> and `MATCHING` from the module created here. See Implementation order.

Satisfies the three preregistration requirements attached to plan approval: **(a)** the exposure formula and the rule fixing it, **(b)** the shipped-only criterion deciding paired `would_clip` versus candidate-only revisit rate, **(c)** all numeric PASS/FAIL criteria — every one written before execution.

**Files:**
- Create: `scripts/GPU/alphazero/v18_preflight_criteria.py`
- Test: `tests/test_v18_preflight_criteria.py`

**Interfaces:**
- Consumes: `canonical_json_bytes`, `fpu_provenance`.
- Produces: `SCOPE_BOUNDARY`, `PRIMARY_EXPOSURE_FORMULA`, `DESCRIPTIVE_EXPOSURE_FORMULAS`, `EXPOSURE_CUTOFF_RULE`, `SIGN_DOMINANCE`, `REACH`, `TERMINAL_FRACTION`, `SEPARATION`, `REVISIT_FORM_CRITERION`, `HISTORICAL_ANCHORS`, `R_MAX_RULE`, `R_MIN_RULE`, `CONVERSION_EFFICIENCY_MIN`, `MIN_LOST_REPLIES`, `STABLE_LEADER_MIN_FRACTION`; `emit_frozen_criteria(path) -> str`.

**(a) One formula, frozen before measurement.** Revision 1's first-past-the-post rule still let selected-A choose the selector by A-vs-control AUC, contradicting "A may only demonstrate reach". The primary statistic is therefore fixed *a priori* to **`contribution_weighted_positive_mass`**, chosen because it is the statistic most directly aligned with the claimed backup-mass mechanism: it weights each positive over-cap residual by the share of root backups that actually flow through that leaf. `positive_count` and `positive_clipped_mass` are computed and reported as **descriptive diagnostics only** and **cannot rescue a failure of the primary**.

**Exact formulas.** Every criterion is mechanical; each names its population and denominator.

```text
exposure(row, cap) = contribution_weighted_positive_mass
    = sum over eligible depth-2 leaves of
        (terminating_backups(leaf) / root.visit_count) * max(0, residual - cap)
      evaluated at the STRONGEST cap 0.50

sign_dominance(row)
    = positive_mass / (positive_mass + negative_mass)
      positive_mass = sum(max(0,  residual_i))
      negative_mass = sum(max(0, -residual_i))
      zero denominator -> 0.0, and the row is ineligible as a target

reach = POOLED over the A rows, not a mean of per-row ratios:
      numerator   = sum over A rows, over eligible leaves with residual > 0.50,
                    of terminating_backups(leaf) * max(0, leaf.nn_value)
      denominator = same sum over ALL eligible leaves
      zero denominator -> PREFLIGHT_FAIL

terminal_fraction = POOLED over A rows and control rows:
      numerator   = depth-2 nodes with visit_count > 0 that are terminal
      denominator = depth-2 nodes with visit_count > 0 (terminal + eligible +
                    synthetic)
      also reported per population

separation AUC:
      row unit    = one position (never one leaf)
      groups      = the 30 A rows (positive class) vs the 30 MATCHED cohort
                    rows (negative class). n_A = n_C = 30 is frozen before
                    measurement, so the operating-characteristic table below is
                    computable and approvable NOW rather than after the census.
      statistic   = PRIMARY_EXPOSURE_COLUMN, i.e. the census column
                    `exposure_primary_0.50` -- named by the shared constant, not
                    by a restated literal
      estimator   = Mann-Whitney U / (n_A * n_C), ties contributing exactly 0.5
      uncertainty = 10,000-replicate stratified bootstrap under frozen seed
                    20260729
      lower bound = the ONE-SIDED 95% lower bound, i.e. the bootstrap quantile
                    at q = 0.05, taken with the protocol's sorted
                    linear-interpolation convention at rank q * (n - 1).
                    Revision 5 wrote rank 0.95 * (n - 1), which is the 95th
                    PERCENTILE -- the UPPER endpoint -- not a lower bound.
                    One-sided q = 0.05 is chosen over two-sided q = 0.025
                    because only the lower endpoint gates, and it matches the
                    one-sided Clopper-Pearson convention used by sizing.
      PASS        = point estimate >= SEPARATION.min_auc
                    AND one-sided 95% lower bound >= 0.5
      weighting   = none; every row counts once
      FAILURE MEANS, frozen now so it cannot be renegotiated after the number is
                    known: "required A-vs-matched-control selectivity was not
                    established". It does NOT mean "no effect exists". The
                    approved operating characteristics show 51.7% power when the
                    true AUC equals the 0.70 threshold, so a miss at the
                    boundary is close to a coin flip and carries no evidential
                    weight against the mechanism. Any downstream verdict text
                    that reads a separation failure as refutation is wrong.

ROLE ASSIGNMENT -- the four roles are an EXCLUSIVE CLASSIFICATION, not four
independent eligibility sets. The predicates overlap: a row can satisfy both the
target exposure cutoff and the flip-control exposure floor, and if the cutoff
were ever <= 0.50 an identity witness could also satisfy it. Frozen:

    assignment order        target -> representative -> {identity, flip}

      1. TARGET      exposure >= EXPOSURE_CUTOFF
                     AND sign_dominance >= 0.80
                     AND near-even and minimum-eligible-leaf rules
                     AND NOT flip_control
                     The explicit NOT keeps flip priority: flip controls are the
                     scarcer role and the one that makes the matched-control
                     gate non-vacuous, so starving them to feed targets is the
                     worse failure.
      2. REPRESENTATIVE  exact phase/side quotas drawn from NON-TARGET rows
                     using ONLY near-even, geometry and canonical ordering
                     (canonical_state_sha1 ascending within phase).
                     Identity and flip eligibility are NOT inspected at this
                     step and are not yet assigned.
      3. IDENTITY and FLIP  assigned from the rows remaining after step 2.
      4. SHORTFALL   any role that cannot be filled STOPS the run.
                     Representative choices are never revisited after residual
                     roles become visible.

    WHY THIS ORDER. Revision 6 assigned representatives LAST, after identity,
    flip and target had each removed rows on residual criteria, then sorted the
    survivors by canonical hash and called the result residual-independent. It
    is not: hash-ordering a residual-conditioned CANDIDATE SET does not make the
    selection independent of residuals. The design requires representatives to
    be chosen independently of residual magnitude *after excluding targets* --
    which permits conditioning on target status and nothing else. Selecting them
    at step 2, before identity and flip exist, is what makes that real. It is
    conservative, and the conservatism is the point.

`test_role_predicates_are_mutually_exclusive_on_every_row` is renamed
`test_role_assignment_is_total_and_exclusive` and asserts the ORDER produces a
partition -- revision 5 asserted mutual exclusivity, which the formulas alone do
not establish.

PROSPECTIVE-TARGET SUBSET FLOOR
    The broad-census subset feeding R_min and revisit density must contain at
    least 16 rows (the development target count). Below that, R_min divides
    0/0 and the revisit density rule is vacuous, so an empty or undersized
    subset is PREFLIGHT_FAIL with reason
    prospective_target_subset_below_floor -- never a silently skipped gate.

POPULATION SPLIT -- which non-A set feeds which quantity.
      The matched cohort is 30 rows, so its top exposure decile is about three
      rows: far too few to derive a threshold from. The two roles are therefore
      served by different non-A populations, both shipped-only and both
      untouched by A:

        matched cohort (30 rows)   -> separation AUC, EXPOSURE_CUTOFF
                                      (matching is what makes these comparable
                                       to A row-for-row)
        broad non-A census         -> prospective-target subset, and from it
                                      R_min and the revisit-form decision
                                      (matching is irrelevant here; population
                                       size is what matters)

R_min:
      population  = the PROSPECTIVE TARGET SUBSET of the BROAD NON-A CENSUS,
                    i.e. census rows with exposure(row, 0.50) >= EXPOSURE_CUTOFF.
                    Never A rows (spec Sec 2.2.3); never all ordinary controls;
                    and not the 30-row matched cohort, whose decile is ~3 rows.
      caps        = the signed crossover table is computed at EVERY grid cap,
                    not only the weakest.
      aggregation = POOLED per cap:
                    pooled(c) = (sum predicted_shipped_replies
                                 - sum predicted_capped_replies)
                                / sum predicted_shipped_replies
      fail rule   = PREFLIGHT_FAIL only if NO grid cap has pooled(c) > 0,
                    reason mechanism_not_predicted_to_act_at_any_cap.
      derivation  = let c* be the WEAKEST cap with pooled(c*) > 0.
                    R_min = max(R_MIN_FLOOR, 0.5 * pooled(c*))
                    R_MIN_FLOOR = 0.01, basis "normative".
      The floor NEVER converts a negative prediction into a positive R_min:
      a nonpositive pooled value at a cap makes that cap ineligible to define
      R_min, it does not get floored up.

      WHY THIS CHANGED. Revision 2 pooled over ALL ordinary controls at the
      weakest cap and rejected v18 on a nonpositive result. That was backwards
      twice. Small or zero reply reduction on ordinary controls is exactly the
      selectivity v18 claims -- penalising it would reward an indiscriminate
      mechanism. And under-reach at the weakest cap is a Sec 7 ADVANCE outcome,
      never a family rejection; only a Sec 10.3 safety failure rejects.

R_max:
      = min(anchor reply_reduction) * 0.5
      = 0.23358985966500678 * 0.5 = 0.11679492983250339
      A deliberately conservative POLICY MARGIN, not an empirical derivation.
      Spec Sec 2.1.1 permits anchors to LOWER R_max and to do nothing else.
      Fails if R_min >= R_max: an empty band is unsatisfiable, not a pass.

revisit-form density:
      population  = the PROSPECTIVE TARGET SUBSET of the BROAD NON-A CENSUS,
                    i.e. census rows with exposure(row, 0.50) >= EXPOSURE_CUTOFF
                    -- the same population as R_min, and for the same reason:
                    the 30-row matched cohort's decile is about three rows
      floor       = the PROSPECTIVE-TARGET SUBSET FLOOR binds here too: a
                    subset of fewer than 16 rows is PREFLIGHT_FAIL with reason
                    prospective_target_subset_below_floor. Refusing only an
                    EMPTY subset would let a 3-row population decide the form of
                    the whole criterion on 2 dense rows, which is exactly the
                    vacuity the floor exists to prevent.
      paired form iff at least 75% of that subset has >= 5 shipped would_clip
                    leaves at the WEAKEST cap 1.25
      otherwise   = candidate_only_floor

EXPOSURE_CUTOFF (the row-selection threshold the future selector needs, which
revision 1 omitted entirely):
      = nearest-rank 0.90 quantile of exposure(row, 0.50) over the 30-row
        MATCHED COHORT only
      nearest-rank = the ceil(0.90 * n)-th smallest value, 1-indexed; no
                     interpolation, so the value is always an observed datum
      target predicate = exposure >= cutoff (ties ADMITTED)
      deterministic ordering for ties and reporting:
                     (canonical_state_sha1, game_idx, position_ply) ascending
```

**Frozen selection predicates.** Revision 2 left these as prose; they are selection rules and must be numeric before any residual is measured.

```text
PHASE_WINDOWS (ply, established buckets)
    opening   0-30
    early_mid 31-60
    midgame   61-90
    late      >= 91

NEAR_EVEN
    abs(root_value_stm) <= 0.30
    Anchored to the observed failure regime: every one of v16a's 15 new
    collapses was near-even, abs(stm) <= 0.28, median 0.03.

MIN_ELIGIBLE_DEPTH2_LEAVES = 50
    A floor against degenerate roots, not a targeting device. Measured, so it
    is applied by the Task 4b matcher, never by Task 4.

BRANCHING
    n_legal is RECORDED as a stratum, never gated (spec Sec 9.2.4:
    "branching bands recorded, not post-hoc gated"). There is no separate
    high-branching predicate.

CANONICAL FIELD CONTRACT -- every predicate names a column of Task 7's frozen
`census_positions.csv` schema and nothing else. The criteria module and the
measurement CLI must not hold two vocabularies for one quantity:

    exposure_primary_0.50    the primary statistic
    sign_dominance
    root_value_stm           SIGNED; near-even applies abs() as a transform
    eligible_depth2_leaves
    would_clip_1.25          count of leaves the weakest cap would clip
    clipped_amount_1.25      total clipped mass at the weakest cap
    would_clip_0.5           count of leaves the strongest cap would clip

Note the deliberate spelling difference: `exposure_primary_0.50` carries two
decimals because that is the frozen column name, while the per-cap triples render
their cap with `str(cap)`, giving `0.5`, `0.75`, `1.0`, `1.25`. Do not "tidy"
either one into the other -- both are load-bearing joins.

**Each name is written ONCE.** `PRIMARY_EXPOSURE_COLUMN = "exposure_primary_0.50"`
is defined a single time and referenced by `CENSUS_SCHEMA`,
`REQUIRED_CENSUS_FIELDS`, the target role predicate, `SEPARATION`,
`EXPOSURE_CUTOFF_RULE` and `classify_role`. A second spelling of a column name is
exactly how a gate comes to reference a column the census never emits, and a
contract stated in six places is six chances to diverge.

The only admissible non-column in any predicate is a **declared transform**:
`MATCHING.derived_variables` holds `abs_root_value_stm = abs(root_value_stm)`,
because the matcher pairs on magnitude while matching `side_to_move` exactly.
Anything else must be a census column under its real name -- `side` is not a
column, `side_to_move` is.

FLIP_CONTROL_EXPOSURE  -- operator is AND, both constants binding
    would_clip_1.25 >= 3
    AND clipped_amount_1.25 >= 0.50
    Exposure at the weakest cap implies exposure at every stronger cap.

IDENTITY_WITNESS
    would_clip_0.5 == 0
    EXACTLY equivalent to max(abs(eligible depth-2 residual)) <= 0.50, because
    the clip rule is STRICT: a leaf clips at 0.50 iff abs(residual) > 0.50, so a
    zero count is precisely the statement that no residual exceeds it. Stated as
    a count because the census emits counts, not a residual maximum -- the
    maximum is not a column and must not be invented as one.

MATCHING (Task 4b)
    cardinality              1:1, EXACTLY n_A = n_C = 30
                             Frozen before measurement so the AUC bootstrap
                             operating characteristic is computable and
                             approvable now. A cohort of 29 is a PREFLIGHT
                             FAILURE, never a smaller cohort.
    algorithm                deterministic minimum-cost bipartite matching
                             (rectangular Hungarian; no scipy in this venv, so
                             vendored)
    feasibility              a pair is admissible only inside every tolerance
                             below; an inadmissible pair has infinite cost and
                             may never appear in a matching
    cost                     sum of per-variable normalized absolute
                             differences, each divided by its own tolerance so
                             all terms are in [0, 1]
    tie-breaking             lexicographic on
                             (cost, control canonical_state_sha1,
                              control game_content_sha1, control position_ply)
    tolerances               keyed by the CENSUS COLUMN name, never an alias
        phase                exact
        side_to_move         exact
        abs_root_value_stm   within 0.10  -- the one DECLARED transform,
                             abs(root_value_stm); the matcher pairs on
                             magnitude and side_to_move carries the direction
        n_legal              within 50
        eligible_depth2_leaves within 40

    Greedy nearest-neighbour is FORBIDDEN: it can fail on A-row ordering even
    when a complete valid matching exists, which would turn a solvable problem
    into a spurious preflight failure.

PER_GAME
    controls: at most 1 position per game (removes within-game correlation,
              so no game-clustered bootstrap is needed)
    future corpus: at most 2 per game, >= 12 plies apart (spec Sec 9.2.4)

UNIVERSE -- fixed independently of which candidate source wins.
    The ORDER of operations is frozen, because "eligible games" could otherwise
    silently come to mean "games with at least one census row", which would
    remove zero-yield games from the sizing universe and bias success upward:

      1. authenticate the complete source
      2. remove FORBIDDEN WHOLE GAMES -- replay CONTENT-SHA exclusions ONLY.
         Canonical-position hashes are POSITION exclusions and belong in step 5.
         Applying them here would drop whole games for holding one forbidden
         position, which silently defeats zero-yield retention.
      3. sort the survivors by replay content SHA-1 ascending and take
         EXACTLY the first 800  -- this fixes all_game_ids
      4. enumerate census positions INSIDE those 800 games
      5. apply POSITION exclusions (canonical hashes), which never remove a game
         from all_game_ids

    n_games                  EXACTLY 800
    if step 3 cannot supply 800 -> STOP
    zero-yield games          RETAINED in all_game_ids. A selected game stays in
                              the sizing universe even if it contributes zero
                              census rows. This matches sizing_analysis_core,
                              whose docstring states all_game_ids is "the
                              COMPLETE reservoir universe ... including games
                              that yielded ZERO kept rows -- excluding them
                              would bias success upward."

    Candidate sources 2 and 3 are larger than 800; truncating them at step 3 by
    this pre-measurement deterministic rule keeps the Task 8 ladder valid
    whichever source is chosen, and caps the census at 4,800 searches.

CENSUS (Task 7 measurement universe)
    positions_per_game       6, PHASE-STRATIFIED as 1 / 1 / 1 / 3
                             opening 1, early_mid 1, midgame 1, late 3
    position rule            within each phase, over that phase's ASCENDING
                             DISTINCT qualifying plies, take the nearest-rank
                             quantile at index ceil(q * n), 1-indexed, without
                             replacement:
                               opening / early_mid / midgame  q = 1/2
                               late                           q = 1/4, 2/4, 3/4
                             Nearest-rank is an INDEX rule, so it is exact and
                             needs no tie convention. Revision 5's "ties to the
                             LOWER ply" described a different, nearest-distance
                             algorithm and is withdrawn.
    missing phases           contribute zero and are REPORTED per game; they are
                             never backfilled from another phase
    max_total_searches       4800 = 800 games x 6   -- hard ceiling; exceeding
                             it aborts before the evaluator loads rather than
                             silently truncating
    ordering                 games by replay content SHA-1 ascending, then ply
                             ascending

    WHY STRATIFIED. Six evenly spaced GLOBAL ply quantiles can omit a narrow
    late interval entirely, or overrepresent late positions in long games, even
    when eligible positions exist in every phase. The selector geometry is
    phase-sensitive, so a phase-biased census biases the sizing estimate.

SIZING (Task 8)
    tier ladder              200, 300, 400, 500, 600, 700 -- probabilistic
                             800                          -- degenerate
    trials per tier          299, for the PROBABILISTIC tiers only
    trial draw               a random subset of that tier's size from the
                             800-game universe, under frozen seed 20260729
    success criterion        an EXACT-SELECTOR witness filling the complete
                             four-role geometry -- never a capacity bound
    tier passes iff          _binomial_lower_bound(k, 299, alpha=0.05) >= 0.99
                             IMPORTED from fpu_dev_corpus_v2 (:3876) -- the
                             exact ONE-SIDED Clopper-Pearson lower bound. Its
                             own docstring records the rule: 299 all-success
                             gives 0.99003 >= 0.99, 298 gives 0.98999. Do not
                             re-derive it, and do not say "95% interval": the
                             quantity is a one-sided lower bound at alpha 0.05.

    TIER 800 IS NOT 299 TRIALS. Drawing 800 games from an 800-game universe
    returns the same set every time, so repeating it 299 times cannot estimate
    a binomial success probability -- the existing v2 implementation correctly
    treats the full-reservoir tier as ONE degenerate trial. Therefore:
      * only tiers 200-700 can qualify probabilistically;
      * tier 800 is a single degenerate exact-selector witness and may serve
        as the next-tier-up operational size;
      * if 700 does not qualify, SIZING FAILS. 299 repetitions of the 800-game
        set cannot rescue it.

    reported size            the smallest probabilistically passing tier, then
                             the next-tier-up margin rule
    reported witness         drawn from a SUCCESSFUL frozen-seed trial at that
                             tier. NOT the content-SHA prefix: a prefix can fail
                             even when every random trial passes, so it cannot
                             be promised as the witness unless prefix success is
                             added as a separate frozen gate. It is not.

    DETERMINISTIC vs RESAMPLED -- these answer different questions:
      * the 299 RANDOM subsets estimate the PROBABILITY that a tier-sized draw
        admits a complete selection;
      * the reported WITNESS is one reproducible successful draw, identified by
        its trial index under the frozen seed.
```

**MIN_LOST_REPLIES is per stage:** development `20`, held-out `30` (scaled with 16 → 24 targets). `STABLE_LEADER_MIN_FRACTION = 0.75` gives 12/16 and 18/24.

**CONVERSION_EFFICIENCY_MIN = 0.5** is a **normative budget-conversion requirement**. Revision 1 justified it by claiming the ratio is "bounded near 1.0" because one un-scanned reply frees exactly one simulation. That reasoning is **withdrawn**: root allocation and stable-leader-subtree traffic both change under the cap, so gained deep backups are not conserved one-for-one against lost replies and the ratio is not bounded that way. The floor stands as a requirement we impose, not a bound we derive.

- [ ] **Step 1: Write the failing test**

Create `tests/test_v18_preflight_criteria.py` covering, each written out in full:

```python
def test_primary_formula_is_frozen_and_single(): ...
def test_descriptive_formulas_cannot_rescue(): ...       # flag present and False
def test_target_selection_never_uses_absolute_residuals(): ...
    # Renamed from test_no_criterion_uses_absolute_residuals. Flip controls
    # LEGITIMATELY use absolute exposure (abs(residual) > 1.25); the prohibition
    # binds TARGET selection only, where Sec 1.3's directional prediction lives.
def test_flip_control_exposure_is_an_AND_of_two_frozen_constants(): ...
def test_every_selection_predicate_is_numeric(): ...     # no prose thresholds
def test_matching_tolerances_are_frozen_for_every_variable(): ...
def test_per_game_caps_differ_for_controls_and_corpus(): ...
def test_exposure_cutoff_rule_is_control_only_with_nearest_rank(): ...
def test_exposure_cutoff_tie_convention_is_admit_and_ordering_is_total(): ...
def test_sign_dominance_formula_names_its_zero_denominator_behaviour(): ...
def test_reach_is_pooled_not_a_mean_of_ratios(): ...
def test_terminal_fraction_names_its_denominator(): ...
def test_separation_declares_row_unit_ties_weighting_and_bootstrap(): ...
def test_r_min_population_is_the_prospective_target_subset(): ...
def test_r_min_is_evaluated_at_every_cap_not_only_the_weakest(): ...
def test_r_min_fails_only_when_no_cap_predicts_positive_conversion(): ...
def test_weak_cap_under_reach_is_an_advance_outcome_not_a_rejection(): ...
def test_r_min_failure_reason_is_the_frozen_string(): ...
    # Task 5 freezes "mechanism_not_predicted_to_act_at_any_cap".
    # Revision 3's verdict test asserted the obsolete
    # "mechanism_not_predicted_to_act"; the two must not drift.
def test_r_min_rule_fails_rather_than_floors_a_nonpositive_prediction(): ...
def test_r_max_is_a_policy_margin_and_strictly_below_every_anchor(): ...
def test_empty_r_band_is_a_failure_not_a_pass(): ...
def test_revisit_density_population_is_the_prospective_target_subset(): ...
def test_min_lost_replies_has_separate_development_and_heldout_values(): ...
def test_conversion_efficiency_is_labelled_normative_not_derived(): ...
def test_historical_anchors_carry_exact_values_and_artifact_sha1s(): ...
def test_emit_frozen_criteria_is_byte_reproducible(tmp_path): ...
def test_frozen_criteria_stamps_scope_boundary_and_forbids_interpretation(): ...

# AUC tail and OC generator -- tested HERE, where they are implemented.
# Revision 6 named the only anti-q=0.95 regression in Task 9, three tasks away
# from the code it guards.
def test_lower_tail_quantile_is_q_0_05(): ...
def test_quantile_interpolation_on_a_known_vector(): ...
    # pin rank q*(n-1) with linear interpolation against hand-computed values,
    # and assert q=0.05 and q=0.95 give DIFFERENT answers on that vector
def test_upper_quantile_cannot_pass_as_the_lower_bound(): ...
def test_gaussian_dgp_delta_mapping_is_sqrt2_probit_of_auc(): ...
    # delta(0.70) == 0.7416, delta(0.80) == 1.1902, delta(0.50) == 0.0
def test_outer_and_bootstrap_streams_are_separate_and_deterministic(): ...
def test_oc_table_is_reproducible_across_two_generations(): ...
def test_oc_generator_reads_no_measurement_artifact(): ...
    # it must be pure simulation: no census, no cohort, no preflight artifact

# Integration with the producer of the rows these criteria classify.
def test_role_assignment_is_total_and_exclusive(): ...
    # named in "Exact formulas" as the rename of
    # test_role_predicates_are_mutually_exclusive_on_every_row; asserts the
    # ORDER produces a partition, which the formulas alone do not establish
def test_classifier_consumes_the_frozen_census_schema(): ...
    # build a row with the EXACT census_positions.csv column names and pass it
    # through classify_role; assert every field the classifier reads is a member
    # of CENSUS_SCHEMA, so the criteria module and Task 7 cannot drift apart
def test_identity_is_would_clip_0_5_equals_zero(): ...
    # equivalent to max|residual| <= 0.50 under the STRICT clip rule
def test_revisit_form_refuses_a_subset_below_the_floor(): ...
    # 15 rows -> ValueError naming prospective_target_subset_below_floor;
    # 12/16 dense -> paired; 11/16 dense -> candidate_only_floor
def test_separation_failure_interpretation_is_frozen(): ...
    # "selectivity not established", never "no effect exists"
def test_primary_exposure_column_is_defined_once_and_used_everywhere(): ...
    # SEPARATION.statistic == EXPOSURE_CUTOFF_RULE.statistic ==
    # PRIMARY_EXPOSURE_COLUMN, which is in CENSUS_SCHEMA; and the retired
    # spelling "exposure_at_cap_0.50" appears nowhere in as_dict()
```

`test_min_lost_replies_has_separate_development_and_heldout_values` must assert `MIN_LOST_REPLIES == {"development": 20, "held_out": 30}`. `test_r_min_rule_fails_rather_than_floors_a_nonpositive_prediction` must assert that the rule dict records `on_nonpositive: "PREFLIGHT_FAIL"` and a reason string, not a floor.

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest -p no:cacheprovider tests/test_v18_preflight_criteria.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the implementation**

Create `scripts/GPU/alphazero/v18_preflight_criteria.py` encoding every formula above as data, with `SCOPE_BOUNDARY = {"mcts_py_edit_authorized": False, "positive_cap_search_authorized": False, "scientific_acceptance_run_authorized": False, "commit_authorized": False, "later_stage_authorized": False}`. Each numeric constant carries its rationale inline and is labelled `"basis": "policy" | "normative" | "measured"` so a reader can tell derived numbers from imposed ones.

`emit_frozen_criteria(path)` writes every constant plus `{"run_kind": "preregistration", "scientific_interpretation_forbidden": True, "spec_revision": 3, "mcts_py_unmodified": True, "git_commit": ..., "worktree_clean": ...}` canonically and returns the SHA-1. **It is not called during implementation** — emission happens in the Execution Phase at a clean HEAD.

- [ ] **Step 4: Run the test to verify it passes, then the full suite**

Run: `.venv/bin/python -m pytest -p no:cacheprovider tests/test_v18_preflight_criteria.py -v` then `-q` for the suite.
Expected: PASS, zero failures.

- [ ] **Step 5: STOP — the user approves the numeric block semantically**

Print the criteria without emitting an artifact:

```bash
.venv/bin/python -c "from scripts.GPU.alphazero import v18_preflight_criteria as C; \
import json; print(json.dumps(C.as_dict(), indent=2, sort_keys=True))"
```

Approval here is **semantic** — the numbers and formulas. The binding artifact is emitted later at a clean HEAD (Execution Phase step 2). The user has already indicated: `R_MAX_SAFETY_FACTOR = 0.5` acceptable **as a policy margin**; `CONVERSION_EFFICIENCY_MIN = 0.5` acceptable **as normative**; sign dominance `0.80`, reach `0.50`, terminal `0.10` and the revisit density rule **plausible but pending exact formulas**, which §"Exact formulas" now supplies.

`SEPARATION.min_auc = 0.70` was previously unapprovable because the cohort size was unknown until after the census. **Freezing cardinality at `n_A = n_C = 30` removes that circularity.**

**The simulation needs a frozen data-generating model.** "True AUC 0.70" does not identify a distribution, and different distributions with the same AUC give materially different power, tie rates and bootstrap behaviour. Frozen before the table is generated:

```text
family              equal-variance Gaussian location shift
                    control ~ N(0, 1), A ~ N(delta, 1)
                    delta = sqrt(2) * Phi^-1(AUC)
                    (so AUC 0.70 -> delta = 0.7416, 0.80 -> 1.1902)
sizes               n_A = n_C = 30
outer datasets      2000 Monte Carlo datasets per AUC point
bootstrap           10,000 stratified replicates per dataset
seeds               outer 20260731, bootstrap 20260732 -- SEPARATE, so the
                    bootstrap stream cannot correlate with dataset generation
percentile          the protocol's sorted linear-interpolation convention at
                    rank q * (n - 1), with q = 0.05 for the one-sided lower
                    bound -- NOT 0.95, which names the upper endpoint
ties                contribute exactly 0.5 to the Mann-Whitney statistic;
                    the continuous family makes exact ties measure-zero, so the
                    tie path is exercised by a separate unit test, not by the
                    table
reported            power at true AUC 0.60 / 0.70 / 0.80 / 0.90, false-pass at
                    0.50, EACH with its Monte Carlo standard error
label               "model-specific operating characteristics under an
                    equal-variance Gaussian shift" -- NOT distribution-free
                    power. Real exposure distributions are skewed and bounded
                    below at zero, so these numbers bound intuition, not truth.
```

Pure simulation, no measurement, so it can be produced now:

```bash
.venv/bin/python -m scripts.GPU.alphazero.v18_preflight_criteria --auc-oc-table
```

Ask for the `0.70` decision only with that table in hand.

- [ ] **Step 6: Request authorization, then commit**

```bash
git add scripts/GPU/alphazero/v18_preflight_criteria.py tests/test_v18_preflight_criteria.py
git commit -m "feat(v18): freeze single exposure formula, cutoff rule, revisit criterion and PASS/FAIL criteria"
```

---

### Task 6: Parameterize the capture and add A/6,400 authentication

Spec §2.2.2 requires a parameterization, not a fork, with the v17 default preserved byte-for-byte. Revision 1 missed that the existing authentication compares against the **400-sim** `probe_black_root_value` in the A CSV, which a 6,400-sim run must — correctly — fail. The 6,400 reference is a different artifact:

```text
logs/eval/v15_budget_check/a_predrop_base_6400sims.csv/position_probe_cases.csv
SHA-1  a17d4737c747e2799253bebbc3d0261e0e697114
mcts_sims 6400, base_seed 20260616, 30 cases,
mean_black_root_value -0.04514575960606817  (the frozen V_REF)
mean_top1_share        0.46407291666666667
```

**Files:**
- Modify: `scripts/GPU/alphazero/capture_v17_abcd_selected_moves.py:37-73, 121-130, 196-200`
- Test: `tests/test_capture_abcd_parameterization.py`

**Interfaces:**
- Produces: `mcts_config(mcts_sims: int = MCTS_SIMS)`; `resolve_gates(s: str) -> tuple[str, ...]`; `build_parser()`; `capture(mode: str = "v17_prechange_abcd", out=OUT) -> dict`; `MODES: dict`; `authenticate_against(...)`; `record_envelope(mode)`; **`build_a6400_reference_bundle(run1_path: str, run2_path: str, out_path: str) -> str`** returning the emitted bundle's SHA-1.

**Task 6 owns the bundle producer.** Revision 8 described it only from Task 9's side, so no task's files, interfaces or tests contained it. It lives here, next to the captures it describes:

```text
artifact_kind         "v18_a6400_reference_bundle"
schema_version        1
capture_run_1_path, capture_run_1_sha1
capture_run_2_path, capture_run_2_sha1
byte_identical                 bool, computed by the builder
historical_source_path, historical_source_sha1
authentication                 per-case value and top-share result, 30 cases
run_kind, scientific_interpretation_forbidden
```

**The bundle does not carry its own digest.** Revision 8 said it records "its own recorded SHA-1", which is impossible without a recursive definition — a file cannot contain the hash of its own complete bytes. `build_a6400_reference_bundle` **returns** the SHA-1 of the bytes it wrote; whoever binds the bundle records that value externally, in the preflight bundle and the Task 9 verdict.

**Where the two test sets live.** Task 6's builder tests prove canonical, atomic, reproducible emission and correctly *computed* fields (six tests, in the block above). The five *tamper* attacks — a bundle claiming `byte_identical: true` over two differing captures, a capture altered after hashing, a fabricated per-case authentication block, a substituted historical-source path, and missing or extra keys — belong primarily to `load_verified_a6400_bundle` in Task 9, because a hand-edited file is what the verifier exists to resist. The builder cannot be attacked by editing its own output.

**Revision 3: the mode self-constrains; the caller does not.** Revision 2 exposed a free `--auth-source`, letting any caller nominate any reference. Instead there are exactly two named modes, and `--mode` is the only switch:

```text
mode "v17_prechange_abcd"   (default; byte-for-byte the existing behavior)
    gates          A, B, C, D
    mcts_sims      400
    auth source    each gate's own 400-sim cases CSV
    output schema  EXACTLY the existing schema -- no new fields

mode "v18_preflight_a6400"
    gates          exactly ("A",)                      -- rejected otherwise
    mcts_sims      exactly 6400                        -- rejected otherwise
    auth source    logs/eval/v15_budget_check/a_predrop_base_6400sims.csv/
                       position_probe_cases.csv
    auth sha1      a17d4737c747e2799253bebbc3d0261e0e697114  -- verified
    batching       (14, 48, 8), asserted
    seed rule      base_seed 20260616, base ^ game_idx ^ position_ply
    case identity  EXACT on (case_id, game_idx, position_ply, side_to_move,
                   replay_path, canonical_state_sha1) -- not merely equal
                   case_id sets
    output schema  the distinct v18 record, with run_kind and
                   scientific_interpretation_forbidden: true
```

**Schema is version-dispatched.** Revision 2 said to add `mcts_sims`, `auth_source` and the gate list to the emitted output; done unconditionally that changes the v17 default bytes and breaks the byte-identity regression it also demands. New fields and labels appear **only** in the v18 mode.

- [ ] **Step 1: Write the failing test**

Create `tests/test_capture_abcd_parameterization.py`:

```python
"""Sec 2.2.2: parameterize for A/6400 WITHOUT changing v17 default behavior.
No GPU: argument plumbing and authentication wiring only."""
import inspect
import pytest

from scripts.GPU.alphazero import capture_v17_abcd_selected_moves as M

SIX_K_REF = ("logs/eval/v15_budget_check/a_predrop_base_6400sims.csv/"
             "position_probe_cases.csv")
SIX_K_REF_SHA1 = "a17d4737c747e2799253bebbc3d0261e0e697114"


def test_v17_defaults_are_unchanged():
    assert M.MCTS_SIMS == 400
    assert (M.EVAL_BATCH_SIZE, M.STALL_FLUSH_SIMS, M.PENDING_VIRTUAL_VISITS) == (14, 48, 8)
    assert set(M.GATES) == {"A", "B", "C", "D"}
    assert M.GATES["A"]["base_seed"] == 20260616
    assert M.GATES["A"]["seed_rule"] == "base ^ game_idx ^ position_ply"


def test_mcts_config_defaults_to_400_and_accepts_an_override():
    assert M.mcts_config().n_simulations == 400
    assert M.mcts_config(6400).n_simulations == 6400


def test_mcts_config_preserves_the_batching_triple_at_any_sim_count():
    cfg = M.mcts_config(6400)
    assert (cfg.eval_batch_size, cfg.stall_flush_sims, cfg.pending_virtual_visits) == (14, 48, 8)


def test_capture_signature_defaults_reproduce_v17_behavior():
    p = inspect.signature(M.capture).parameters
    assert p["mode"].default == "v17_prechange_abcd"


def test_cli_exposes_only_mode_and_out():
    ns = M.build_parser().parse_args([])
    assert ns.mode == "v17_prechange_abcd" and ns.out == M.OUT
    # No caller-nominated reference: the mode fixes every scientific parameter.
    assert not hasattr(ns, "auth_source")
    assert not hasattr(ns, "mcts_sims")
    assert not hasattr(ns, "gates")


def test_v18_mode_is_fully_self_constrained():
    m = M.MODES["v18_preflight_a6400"]
    assert m["gates"] == ("A",)
    assert m["mcts_sims"] == 6400
    assert m["auth_source"] == SIX_K_REF
    assert m["auth_sha1"] == SIX_K_REF_SHA1
    assert m["base_seed"] == 20260616
    assert m["seed_rule"] == "base ^ game_idx ^ position_ply"
    assert m["batching"] == (14, 48, 8)


def test_default_mode_emits_the_existing_schema_with_no_new_keys():
    """Adding mcts_sims / auth_source / gate-list fields unconditionally would
    change the v17 default bytes and break its own byte-identity regression."""
    v17_keys = set(M.document_keys("v17_prechange_abcd"))
    v18_keys = set(M.document_keys("v18_preflight_a6400"))
    assert v17_keys == set(M.LEGACY_DOCUMENT_KEYS)
    assert v18_keys - v17_keys                      # new fields exist...
    assert not (v17_keys - set(M.LEGACY_DOCUMENT_KEYS))   # ...but only in v18


def test_unknown_mode_rejected():
    with pytest.raises((SystemExit, ValueError, KeyError)):
        M.capture(mode="whatever")


def test_case_identity_is_the_full_tuple_not_just_case_id():
    # Same case_id, different ply: equal case_id SETS would have passed.
    with pytest.raises(ValueError, match="case set"):
        M.authenticate_against(source_rows=[_src()],
                               captured=[_cap(position_ply=9)])


# --- A/6,400 reference bundle: BUILDER tests --------------------------------
# The builder's job is correct, canonical, atomic emission and correctly
# COMPUTED fields. Tamper attacks live with load_verified_a6400_bundle in
# Task 9, which is what must resist a hand-edited file.

# The frozen key set, written out INDEPENDENTLY of the implementation. Comparing
# against M.A6400_BUNDLE_KEYS would let an added key plus an edited tuple pass.
EXPECTED_BUNDLE_KEYS = {
    "artifact_kind", "schema_version",
    "capture_run_1_path", "capture_run_1_sha1",
    "capture_run_2_path", "capture_run_2_sha1",
    "byte_identical",
    "historical_source_path", "historical_source_sha1",
    "authentication",
    "run_kind", "scientific_interpretation_forbidden",
}


# Tracked 30-case fixture, committed under
#   tests/golden/a6400_bundle_fixture/{source_rows.json,capture.json}
# A clean checkout cannot derive these identities from the real artifact: the
# frozen A cases CSV lives under gitignored logs/. Revision 12 called
# `M.testing.synthetic_capture`, a test-only namespace that does not exist and
# must not be added to a production module -- fixture construction belongs here.
FIXTURE_DIR = Path(__file__).parent / "golden" / "a6400_bundle_fixture"


def _fixture(name):
    return json.loads((FIXTURE_DIR / name).read_text())


@pytest.fixture
def source_rows():
    rows = _fixture("source_rows.json")
    assert len(rows) == 30
    return rows


@pytest.fixture
def capture_doc():
    """One capture document carrying the full 30-case set, matching source_rows
    so `authenticate_against`'s exact-30 rule is satisfiable."""
    doc = _fixture("capture.json")
    assert len(doc["cases"]) == 30
    return doc


@pytest.fixture
def captures(tmp_path, capture_doc):
    """Two byte-identical captures plus a third that genuinely differs."""
    import json
    other = json.loads(json.dumps(capture_doc))
    other["cases"][0]["top_share_repr"] = "0.99"
    run1, run2, run2d = (tmp_path / "r1.json", tmp_path / "r2.json",
                         tmp_path / "r2d.json")
    for p, d in ((run1, capture_doc), (run2, capture_doc), (run2d, other)):
        p.write_text(json.dumps(d, sort_keys=True))
    return str(run1), str(run2), str(run2d)


@pytest.fixture
def frozen_source(monkeypatch, source_rows):
    """Point the wrapper's frozen historical source loader at the fixture, so
    the 30-case authentication succeeds without the real 6,400 artifact."""
    monkeypatch.setattr(M, "_load_frozen_a6400_source", lambda: source_rows)


def test_frozen_source_loader_opens_exactly_the_frozen_path(monkeypatch,
                                                            source_rows):
    """BEHAVIORAL, not documentary.

    Revision 13 asserted the path appeared in the loader's DOCSTRING, which a
    loader reading a byte-identical copy from anywhere else would also pass.
    Instrument the real read/hash boundary and assert the path actually used.
    """
    frozen = M.MODES["v18_preflight_a6400"]["auth_source"]
    raw = b'[{"case_id": "x"}]'
    seen = []
    monkeypatch.setattr(M, "_read_bytes",
                        lambda p: (seen.append(("read", p)), raw)[1])
    monkeypatch.setattr(M, "sha1_bytes",
                        lambda b: (seen.append(("hash", b)),
                                   M.MODES["v18_preflight_a6400"]["auth_sha1"])[1])
    monkeypatch.setattr(M, "_parse_source_rows",
                        lambda b: (seen.append(("parse", b)), source_rows)[1])

    assert M._load_frozen_a6400_source() == source_rows

    # Exactly one read, of the frozen path, then hash, then parse.
    assert [kind for kind, _ in seen] == ["read", "hash", "parse"]
    assert seen[0][1] == frozen
    # THE decisive assertion: the hashed object and the parsed object are the
    # SAME bytes, not two reads that happened to agree. Revision 15 hashed a
    # path and parsed a path, so a file changed in between would authenticate
    # one sequence and parse another -- ordering alone could not catch it.
    assert seen[1][1] is raw
    assert seen[2][1] is raw


def test_frozen_source_loader_refuses_a_hash_mismatch_without_parsing(
        monkeypatch, source_rows):
    parses = []
    monkeypatch.setattr(M, "_read_bytes", lambda p: b"whatever")
    monkeypatch.setattr(M, "sha1_bytes", lambda b: "0" * 40)
    monkeypatch.setattr(M, "_parse_source_rows",
                        lambda b: (parses.append(b), source_rows)[1])
    with pytest.raises(ValueError, match="a17d4737"):
        M._load_frozen_a6400_source()
    assert parses == []         # authentication precedes any parse


def test_frozen_source_loader_takes_no_path_argument():
    """Path substitution must be impossible by construction: the frozen path is
    not a parameter."""
    import inspect
    assert inspect.signature(M._load_frozen_a6400_source).parameters == {}


def test_bundle_emission_is_canonical_and_byte_reproducible(tmp_path, captures, frozen_source):
    run1, run2, _ = captures
    a = M.build_a6400_reference_bundle(run1, run2, str(tmp_path / "a.json"))
    b = M.build_a6400_reference_bundle(run1, run2, str(tmp_path / "b.json"))
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()
    assert a == b


def test_builder_returns_the_sha1_of_the_bytes_it_wrote(tmp_path, captures, frozen_source):
    import hashlib
    run1, run2, _ = captures
    p = tmp_path / "bundle.json"
    returned = M.build_a6400_reference_bundle(run1, run2, str(p))
    assert returned == hashlib.sha1(p.read_bytes()).hexdigest()


def test_bundle_never_contains_its_own_digest(tmp_path, captures, frozen_source):
    """A file cannot carry the hash of its own complete bytes."""
    run1, run2, _ = captures
    p = tmp_path / "bundle.json"
    sha = M.build_a6400_reference_bundle(run1, run2, str(p))
    assert sha not in p.read_text()


def test_builder_refuses_differing_captures_and_writes_nothing(
        tmp_path, captures, frozen_source):
    """A reference bundle over two DIFFERENT captures is not a valid artifact:
    its single `authentication` block would be undefined as to which run it
    describes. Refuse rather than record byte_identical False."""
    run1, _run2, run2_different = captures
    p = tmp_path / "bundle.json"
    with pytest.raises(ValueError, match="byte-identical"):
        M.build_a6400_reference_bundle(run1, run2_different, str(p))
    assert not p.exists()


def test_builder_refuses_when_authentication_fails_and_writes_nothing(
        tmp_path, captures, source_rows, monkeypatch):
    import copy
    run1, run2, _ = captures
    bad = copy.deepcopy(source_rows)
    bad[0]["probe_black_root_value"] = "0.99"
    monkeypatch.setattr(M, "_load_frozen_a6400_source", lambda: bad)
    p = tmp_path / "bundle.json"
    with pytest.raises(ValueError):
        M.build_a6400_reference_bundle(run1, run2, str(p))
    assert not p.exists()


def test_authentication_report_is_returned_and_stored_once(
        tmp_path, captures, source_rows, capture_doc, frozen_source):
    """The bundle's 30-entry block is exactly what authenticate_against returns,
    and both captures must produce canonically identical reports."""
    import json as _json
    run1, run2, _ = captures
    p = tmp_path / "bundle.json"
    M.build_a6400_reference_bundle(run1, run2, str(p))
    expected = M.authenticate_against(source_rows, capture_doc["cases"])
    assert _json.loads(p.read_text())["authentication"] == expected


def test_byte_identical_is_computed_not_copied(tmp_path, captures, frozen_source):
    """On the valid path the field is an INVARIANT, and the builder computes it
    rather than accepting any claim."""
    import json
    run1, run2, _ = captures
    p = tmp_path / "bundle.json"
    M.build_a6400_reference_bundle(run1, run2, str(p))
    assert json.loads(p.read_text())["byte_identical"] is True


def test_authentication_block_covers_all_thirty_cases(tmp_path, captures,
                                                      frozen_source):
    import json
    run1, run2, _ = captures
    p = tmp_path / "bundle.json"
    M.build_a6400_reference_bundle(run1, run2, str(p))
    assert len(json.loads(p.read_text())["authentication"]) == 30


def test_bundle_has_the_exact_frozen_key_set(tmp_path, captures, frozen_source):
    import json
    run1, run2, _ = captures
    p = tmp_path / "bundle.json"
    M.build_a6400_reference_bundle(run1, run2, str(p))
    # Compared against the INDEPENDENT literal above, not M.A6400_BUNDLE_KEYS.
    assert set(json.loads(p.read_text())) == EXPECTED_BUNDLE_KEYS


def test_implementation_key_tuple_matches_the_independent_literal():
    assert set(M.A6400_BUNDLE_KEYS) == EXPECTED_BUNDLE_KEYS


def test_bundle_emission_is_atomic(tmp_path, captures, frozen_source, monkeypatch):
    """Failure AFTER the temp file exists must leave the destination absent and
    no temp file behind.

    Revision 10's version passed a nonexistent input, which fails while opening
    it -- before any output write is attempted -- so it proved nothing about
    atomicity. Force the failure at the rename instead.
    """
    import os
    run1, run2, _ = captures
    p = tmp_path / "bundle.json"

    def boom(src, dst):
        raise OSError("simulated failure during replace")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        M.build_a6400_reference_bundle(run1, run2, str(p))
    assert not p.exists()
    assert list(tmp_path.glob("*.tmp*")) == []


def test_atomic_write_leaves_an_existing_destination_unchanged(
        tmp_path, captures, frozen_source, monkeypatch):
    import os
    run1, run2, _ = captures
    p = tmp_path / "bundle.json"
    p.write_text("PRIOR CONTENT")
    monkeypatch.setattr(os, "replace",
                        lambda s, d: (_ for _ in ()).throw(OSError("boom")))
    with pytest.raises(OSError):
        M.build_a6400_reference_bundle(run1, run2, str(p))
    assert p.read_text() == "PRIOR CONTENT"


def test_unknown_gate_rejected():
    with pytest.raises((SystemExit, ValueError, KeyError)):
        M.resolve_gates("A,Z")


def test_auth_source_case_set_must_match_exactly():
    """Authentication compares against a DIFFERENT artifact, so a case-set
    mismatch must abort rather than authenticate a subset."""
    with pytest.raises(ValueError, match="case set"):
        M.authenticate_against(
            source_rows=[{"case_id": "x", "probe_black_root_value": "0.0",
                          "probe_top1_share": "0.5"}],
            captured=[{"case_id": "y", "recomputed_black_value_repr": "0.0",
                       "top_share_repr": "0.5"}])


def _src(**over):
    """A source row carrying the FULL six-field identity, so an authentication
    test fails on the condition it names rather than on a missing key."""
    row = {"case_id": "x", "game_idx": 1, "position_ply": 7,
           "side_to_move": "black", "replay_path": "p",
           "canonical_state_sha1": "a" * 40,
           "probe_black_root_value": "0.25", "probe_top1_share": "0.50"}
    row.update(over)
    return row


def _cap(**over):
    row = {"case_id": "x", "game_idx": 1, "position_ply": 7,
           "side_to_move": "black", "replay_path": "p",
           "canonical_state_sha1": "a" * 40,
           "recomputed_black_value_repr": "0.25", "top_share_repr": "0.50"}
    row.update(over)
    return row


def test_auth_checks_value_and_top_share_not_value_alone():
    with pytest.raises(ValueError, match="top_share"):
        M.authenticate_against(source_rows=[_src()],
                               captured=[_cap(top_share_repr="0.99")])


def test_auth_checks_value_too():
    with pytest.raises(ValueError, match="value"):
        M.authenticate_against(source_rows=[_src()],
                               captured=[_cap(recomputed_black_value_repr="0.99")])


def test_auth_passes_when_both_statistics_and_full_identity_match():
    M.authenticate_against(source_rows=[_src()], captured=[_cap()])


def test_record_kind_is_stamped_and_interpretation_forbidden():
    assert M.record_envelope("v18_preflight_a6400")["run_kind"] == "v18_preflight_a6400"
    assert M.record_envelope("v18_preflight_a6400")[
        "scientific_interpretation_forbidden"] is True
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest -p no:cacheprovider tests/test_capture_abcd_parameterization.py -v`
Expected: FAIL — `mcts_config()` takes no arguments; `build_parser`, `resolve_gates`, `authenticate_against`, `record_envelope` do not exist.

- [ ] **Step 3: Make the minimal edits**

- `mcts_config(mcts_sims: int = MCTS_SIMS)` passing `mcts_sims` into `EvalConfig`. **Keep the batching assertion at line 123 exactly as it is.**
- `MODES` as the table above; `capture(mode, out)` reads every parameter from the mode and **rejects any deviation**, so no caller can nominate a reference.
- In the default mode the emitted document must be **byte-identical** to today's: build it from the existing code path with no added keys.
- `authentication_report(source_rows, captured) -> list[dict]` — one entry per case, in `case_id` ascending order, carrying the identity tuple, both compared statistics, their absolute differences and a per-case `ok` flag. Pure and total: it reports, it does not raise.
- `authenticate_against(source_rows, captured) -> list[dict]` — require exact identity on the full tuple `(case_id, game_idx, position_ply, side_to_move, replay_path, canonical_state_sha1)`, raising `ValueError` mentioning "case set" on any difference; then compare the recomputed black value to `probe_black_root_value` and the captured top share to `probe_top1_share`, both within `TOLERANCE = 1e-6`, naming "top_share" when that is the mismatching statistic; and on success **return `authentication_report(...)`**.

  Revision 12 specified `authenticate_against` as exact-or-raise returning nothing, while requiring the bundle to store a 30-entry authentication block — leaving the block's provenance undefined. Returning the report closes that: the builder authenticates **both** captures, and because their bytes are byte-identical it **requires the two reports to be canonically identical** and stores one. A divergence there would mean authentication is not a pure function of the capture bytes, which is itself a defect worth raising on.
- Verify the auth source's SHA-1 against the mode's frozen value before reading it.
- `record_envelope(mode)` returning `{"run_kind": ..., "scientific_interpretation_forbidden": True}` for the v18 mode only.
- `resolve_gates`, `build_parser` with `--mode` and `--out` only.
- **`authentication_report(source_rows, captured) -> list[dict]`**; **`build_a6400_bundle_document(...) -> dict`** (pure); **`build_a6400_reference_bundle(run1_path, run2_path, out_path) -> str`** (the production wrapper); **`_load_frozen_a6400_source() -> list[dict]`**, which takes **no arguments** — the frozen path is not a parameter, so substitution is impossible by construction — and authenticates **the bytes it parses**:

```text
raw  = _read_bytes(MODES["v18_preflight_a6400"]["auth_source"])   # the ONLY read
       sha1_bytes(raw) must equal a17d4737c747e2799253bebbc3d0261e0e697114
rows = _parse_source_rows(raw)                                    # the SAME object
```

Revision 15 hashed via `file_sha1(path)` and then parsed via `_read_source_rows(path)` — two independent reads, so a file changed in between would have one byte sequence authenticated and a different one parsed. Hash-before-parse ordering does not fix that; only hashing and parsing the *same in-memory object* does.

Helpers `_read_bytes(path) -> bytes`, `sha1_bytes(raw) -> str` and `_parse_source_rows(raw) -> list[dict]` exist as named seams so a test can observe both the path opened and the object identity threaded through. Plus the frozen key tuple `A6400_BUNDLE_KEYS`.

**The builder accepts only byte-identical, fully authenticated captures.** Revision 11 had it emit `byte_identical: false` for differing captures, which contradicted two frozen rules at once: `authenticate_against` requires the exact 30-case set and raises on mismatch, so a one-case fixture could never satisfy the positive tests; and a bundle with a single `authentication` block is undefined as to *which* run it authenticates when the two differ. Frozen contract:

```text
build_a6400_reference_bundle(run1, run2, out)
    1. load both captures
    2. REQUIRE byte-identical -> else raise, write nothing
    3. authenticate BOTH against all 30 historical cases -> else raise,
       write nothing
    4. build the document, write atomically and canonically
    5. return the SHA-1 of the bytes written; never embed it

`byte_identical` remains a recorded field, but it is now an INVARIANT of a
valid bundle rather than a variable. The verifier still recomputes it
independently -- it must resist a hand-edited file, not trust the builder.
```

A record of a *failed* duplicate run is still useful operationally, but it is **not** a reference bundle: emit it under a separate, explicitly non-binding artifact kind that no verifier accepts.

**Split for testability.** `build_a6400_bundle_document` is pure — it takes already-loaded, already-authenticated inputs and returns the dict — so canonical/reproducible/key-set/no-self-digest tests need no 30-case corpus. The wrapper's refusal and authentication behaviour is tested by monkeypatching its frozen source loader, or against a tracked 30-case fixture where one is affordable.

The wrapper must: reopen and hash both captures; compare their canonical bytes itself; reopen and hash the historical source; run `authenticate_against` over all 30 cases; write **atomically** (temp file plus rename, so a mid-write failure leaves nothing behind) and **canonically** via `canonical_json_bytes`.

- [ ] **Step 4: Run the test to verify it passes, then the full suite**

Expected: PASS, zero failures.

- [ ] **Step 5: Request authorization, then commit**

```bash
git add scripts/GPU/alphazero/capture_v17_abcd_selected_moves.py tests/test_capture_abcd_parameterization.py
git commit -m "feat(v18): parameterize A/B/C/D capture for sims, gate subset and authentication source"
```

---

### Task 7: The preflight measurement CLI

**Files:**
- Create: `scripts/GPU/alphazero/diagnose_v18_residual_preflight.py`
- Test: `tests/test_v18_residual_preflight.py`

**Interfaces:**
- Consumes: `v18_tree_walk.walk`, `v18_crossover.{crossover_for_tree, assert_synchronous_tree}`, `v18_preflight_criteria`, `v18_control_pool` (the frozen record), `eval_runner`, `position_probe_cases.position_state`, `mcts.MCTS.search_with_root`, `fpu_provenance`, `canonical_json_bytes`.
- Produces: `build_row(root, caps, c_puct, meta) -> dict`; `run_preflight(...) -> dict`; artifacts `residual_rows.csv`, `crossover_tables.csv`, `preflight_artifact.json`. **Computes no verdict.**

Populations, both at 400 sims, `add_noise=False`, checkpoint `checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors`.

**Separate frozen base seeds per population.** `game_idx` is reservoir-local, so a single base seed would let an A game and a census game with the same index collide onto the same seed. Frozen:

```text
selected_a   base 20260616, seed = base ^ game_idx ^ position_ply
             (unchanged -- this is the historical A rule and must not move)
census       base 20260730, seed = base ^ game_idx ^ position_ply
```

**The audit compares complete derived seed SETS, not base seeds.** XOR with
reservoir-local `game_idx` and `position_ply` can collide across two different
bases, so comparing `20260616` against `20260730` proves nothing. Before
measurement, materialise every derived seed for selected-A and for the census,
assert the two sets are disjoint, and record both sets' sizes and their
intersection size (which must be zero) in the artifact. Also check them against
the historical derived sets where those are reconstructible.

**A collision is a pre-search STOP, not an in-flight fix.** Revision 5 said to
change the census base and re-audit; by Execution step 3 that base is already
committed and embedded in the emitted criteria, so mutating it would rewrite a
frozen execution. On a nonempty intersection the run aborts **before any
evaluator call**. Changing the base then requires a criteria amendment, updated
tests, a commit, a clean tree, and a restart from Execution step 1. A test must
prove the abort reaches zero evaluator calls.

| population | source | role |
|---|---|---|
| `selected_a` | `logs/eval/calib020_0001_black_loss_post_opening_predrop_probe/position_probe_cases.csv`, rows with `checkpoint == "0001"` (30 positions) | **reach and separation only** |
| `census` | the frozen Task 4 source universe, per-game census | feeds **both** the Task 4b matcher and Task 8 sizing |

Every row is tagged with `population` so Task 9 can enforce the boundary mechanically. The `non_a_control` cohort is a **subset of `census`** chosen by Task 4b after this measurement; it is not a separate search.

**The census is a per-game census, not a sample.** Task 8 estimates tier-by-tier selector yield, which requires knowing how many qualifying positions each game supplies — so the census enumerates several positions per game under the Task 4 predicates, and the cohort later takes at most one per game. Measuring once for both purposes is why this is a single pass; sizing it as if only the cohort mattered would under-measure by roughly the census-per-game factor.

- [ ] **Step 1: Write the failing test**

Create `tests/test_v18_residual_preflight.py` using `tests/fpu_search_fixture.py`'s fake evaluator so no GPU is needed. Write each of these out in full with real assertions:

```python
def test_build_row_tags_population(): ...
def test_build_row_asserts_synchronous_provenance(): ...
def test_measurement_routes_through_search_with_root_and_never_search_from_root(): ...
def test_search_execution_mode_is_not_accepted_from_caller_metadata(): ...
def test_valid_run_passes_the_constant_and_labels_the_artifact_with_it(): ...
def test_a_mutated_mode_constant_refuses_and_writes_no_artifact(): ...
def test_row_carries_per_cap_exposure_for_every_grid_cap(): ...
def test_row_carries_the_primary_and_both_descriptive_formulas(): ...
def test_artifact_is_byte_reproducible_across_two_runs(): ...
def test_artifact_stamps_run_kind_scope_boundary_and_forbidden_flag(): ...
def test_artifact_binds_the_criteria_sha1_and_the_universe_sha1(): ...
def test_census_positions_csv_carries_the_full_schema(): ...
def test_selected_a_and_census_use_different_base_seeds(): ...
def test_cli_refuses_when_the_frozen_criteria_artifact_is_missing(): ...
def test_cli_refuses_when_the_criteria_sha1_does_not_match(): ...
def test_cli_refuses_a_nonzero_cap_search_configuration(): ...
def test_cli_refuses_on_a_dirty_worktree(): ...
def test_cli_emits_no_verdict_key(): ...
```

The five refusal tests are the non-vacuity evidence: each must fail if its guard is removed. `test_cli_emits_no_verdict_key` asserts no key matching `verdict|pass|fail|selected_formula|r_min|r_max` appears in the artifact — the measurement/judgement separation, enforced rather than described.

**The three routing tests are the evidence boundary for synchronous provenance.** `assert_synchronous_tree` can only check the mode it is handed; what makes that mode true is the route this module takes, so the route is what must be tested:

- `test_measurement_routes_through_search_with_root_and_never_search_from_root` — monkeypatch **both** `MCTS.search_with_root` and `MCTS.search_from_root` with recording wrappers, run the measurement over a small case set on the fake evaluator, then assert `search_from_root` recorded **exactly zero** calls and `search_with_root` recorded one per measured position. Zero calls is the load-bearing half: a test that only confirms `search_with_root` was used would still pass a module that called both. **This dynamic observation is the provenance evidence.** A static assertion that the module source never mentions `search_from_root` may be added as a supplemental tripwire against a future second route, but it is not evidence — source text does not establish which path executed, and it must never be offered in place of the zero-call assertion.
- `test_search_execution_mode_is_not_accepted_from_caller_metadata` — `build_row` must **raise** when `meta` carries a `search_execution_mode` key, rather than honouring or ignoring it. A caller-supplied mode is an unaudited claim about someone else's route; accepting one would let a batched measurement label itself synchronous.
- `test_valid_run_passes_the_constant_and_labels_the_artifact_with_it` — spy on `assert_synchronous_tree` and assert **every** call received `M.SEARCH_EXECUTION_MODE` (the constant object, compared against the module attribute, not against a `"synchronous"` literal restated in the test), then assert the emitted artifact's `search_execution_mode` field equals that same constant. This is what ties the asserted mode and the published label to one source on the path that actually produces an artifact.
- `test_a_mutated_mode_constant_refuses_and_writes_no_artifact` — set `SEARCH_EXECUTION_MODE` to `"batched_waiter"` and assert the **full measurement raises `ValueError` and writes no artifact file**, preferably refusing before any search runs. **It must not emit a relabelled artifact.** An earlier draft of this test asserted the artifact label "moves with" the mutated constant; that is self-defeating — `assert_synchronous_tree` rejects any mode but `"synchronous"`, so a successful emission under a mutated constant could only mean the assertion was bypassed, and the control would be certifying the very failure it exists to catch. An invalid route label must produce **no artifact**, never a valid artifact carrying an invalid label. Assert the output path does not exist afterwards, so a partially written file cannot pass. There is no CLI flag that can set the mode; a test asserts the argument parser exposes no such option.

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest -p no:cacheprovider tests/test_v18_residual_preflight.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the implementation**

Requirements:

- Load the frozen criteria artifact and the **frozen universe record**; refuse unless each SHA-1 matches the value **re-derived from the committed module**, not merely a value stored beside the file.
- **One search route, one mode constant.** The module defines `SEARCH_EXECUTION_MODE = "synchronous"` immediately beside its **single** `MCTS.search_with_root` call site, with a comment binding the two: the constant is true *because* of that call, and any edit to the route must move the constant. `search_from_root` is never imported, referenced or called. The constant is the sole source of the mode everywhere downstream — the `assert_synchronous_tree` argument and the artifact label both read it, so they cannot disagree.
- **The mode is never accepted from outside.** No CLI flag sets it, and `build_row` raises `ValueError` if `meta` contains a `search_execution_mode` key. A mode supplied by a caller is an unaudited claim about a route this module did not take.
- Per row: reconstruct the state, run **shipped** `search_with_root`, `assert_synchronous_tree(root, 400, search_execution_mode=SEARCH_EXECUTION_MODE)`, then `walk(root, CAP_GRID)` and `crossover_for_tree(root, cap, c_puct)` per cap.
- `census_positions.csv`: **one row per measured position** — the artifact the matcher and the sizing join both need, absent from revision 3. Exact schema:

```text
population                 selected_a | census
source_universe_ordinal    int, the content-SHA-ascending rank (census only)
game_content_sha1          replay content SHA-1 -- the ONLY game identity
game_idx                   recorded for provenance, never used as identity
position_ply               int
side_to_move               black | red
canonical_state_sha1       canonical position identity
phase                      opening | early_mid | midgame | late
root_value_stm             float, REMEASURED under calib020_0001
n_legal                    int
eligible_depth2_leaves     int
replies, explored_replies, depth_ge3_backups, depth_ge3_fraction
follow_up_visits_per_reply
positive_mass, negative_mass, sign_dominance
terminal_depth2, total_depth2
exposure_primary_0.50      contribution_weighted_positive_mass
exposure_descriptive_count, exposure_descriptive_clipped_mass
would_clip_<cap>, clipped_amount_<cap>, revisit_to_depth3_rate_<cap>
                           one triple per grid cap
seed                       the per-position seed actually used
```

  Task 4b joins on the matching variables; Task 8 joins on
  `(game_content_sha1, position_ply)`.

- `residual_rows.csv`: one row per eligible depth-2 leaf — `population`, `case_id`, `game_idx`, `position_ply`, `side_to_move`, `canonical_state_sha1`, `raw_parent`, `raw_leaf`, `residual`, `leaf_visit_count`, `leaf_terminating_backups`, `leaf_has_depth3_child`, plus `would_clip_<cap>` and `clipped_amount_<cap>` per grid cap.
- `crossover_tables.csv`: one row per (case, cap) with `predicted_shipped_replies`, `predicted_capped_replies`, `predicted_reply_delta`, `predicted_reply_reduction` (**signed, never clamped**), and the exclusion counts.
- `preflight_artifact.json`: per-case `walk` records, the primary and both descriptive exposure values, sign dominance, the pooled reach numerator/denominator, terminal counts, all source SHA-1s, git commit, worktree state, `search_execution_mode` (emitted from `SEARCH_EXECUTION_MODE`, never a literal at the emission site and never caller-supplied), `run_kind: "shipped_only_preflight"`, `scientific_interpretation_forbidden: true`, the criteria SHA-1, the universe-record SHA-1, and `SCOPE_BOUNDARY` verbatim.
- **No verdict, no PASS/FAIL, no derived threshold.**

- [ ] **Step 4: Run the test to verify it passes, then the full suite**

Expected: PASS, zero failures. **Do not run the CLI against real data in this task** — see the Execution Phase.

- [ ] **Step 5: Request authorization, then commit**

```bash
git add scripts/GPU/alphazero/diagnose_v18_residual_preflight.py tests/test_v18_residual_preflight.py
git commit -m "feat(v18): shipped-only residual preflight measurement CLI"
```

---

### Task 8: Residual screen telemetry and exact-selector sizing

Spec §2.2 places selector feasibility and reservoir sizing **inside** the permitted preflight, and §2.3 requires exact selector counts and operational sizes before freeze. Revision 1 omitted this while still producing a final PASS. This task supplies it; without it Task 9's verdict is only provisional.

**Files:**
- Create: `scripts/GPU/alphazero/v18_selector_sizing.py`
- **Modify: `scripts/GPU/alphazero/fpu_dev_corpus_v2.py`** — the narrow version-dispatched hook
- Test: `tests/test_v18_selector_sizing.py`
- Test: `tests/test_fpu_dev_corpus_v2_v18_hook.py`

**`sizing_analysis_core` is not role-neutral and cannot be imported unchanged.** It hard-calls `post_screen_qualification_report(sub_kept, alloc)` and `sample_v2_rows(sub_kept, seed=..., alloc=alloc)`, and iterates `rep["late_target_bands"]` — all specific to v2's two roles (`_ROLES = ("target", "control")`) and its late-target band floors. Revision 5 imported it as though it already accepted four v18 roles. What it *is* safe to import unchanged is `_binomial_lower_bound`.

Per design §5.1 this needs a **narrow version-dispatched hook**, not a fork. Revision 6 left "callbacks **or** schema dispatch" open, which permits two materially different implementations. **One route is now frozen — schema-dispatched role vocabulary and allocation, reusing the existing qualification and exact selector:**

**The schema contract, frozen.** Revision 7 said "roles and allocation become schema-dispatched" without settling what the schema *is*. The selector also hardcodes more than roles: `AllocationProfile.corpus_size`, `split_totals` and `quota_by_phase` all embed the `"tuning"` / `"frozen_check"` literals (`fpu_dev_corpus_v2.py:265-296`), `SPLIT_ALLOC_V2` is v17's 240-row profile (`:195`), `SPLITS = ("tuning", "frozen_check")` is imported from `build_fpu_dev_corpus.py:96`, and `late_target_bands` is consumed unconditionally (`:3927`). Frozen:

```text
schema_version           5
                         NOT 4. Revision 8 claimed the current maximum was 3;
                         that was wrong. `parse_allocation_profile` accepts
                         (2, 3, 4) at :337, and the real v17 development config
                         logs/eval/fpu_v17_baseline_policy_mass/development/
                         fpu_dev_corpus_v2_config.json IS schema 4. Reusing 4
                         would let v18 semantics collide with authenticated v17
                         artifacts. The parser's accepted set becomes
                         (2, 3, 4, 5), and 5 is v18-only.

split vocabulary         ("all",)  -- a SINGLE split.
                         v18's development corpus has no tuning/frozen_check
                         division: spec Sec 11 replaces the frozen split with a
                         separate fresh held-out reservoir. "Whole-game split
                         isolation" therefore means isolation BETWEEN the
                         development and held-out corpora, not within one.

role vocabulary          ("target", "identity_witness", "flip_control",
                          "representative")

allocation (40 rows)     ("target",         "late"):      {"all": 16}
                         ("identity_witness","late"):     {"all":  4}
                         ("flip_control",   "late"):      {"all":  4}
                         ("representative", "opening"):   {"all":  4}
                         ("representative", "early_mid"): {"all":  4}
                         ("representative", "midgame"):   {"all":  4}
                         ("representative", "late"):      {"all":  4}
                         Targets, identity witnesses and flip controls are
                         late-only per spec Sec 9.2.1; only representatives are
                         phase-balanced.

late_target_bands        under schema 5 the qualification report still EMITS the
                         key, with an EMPTY dict, and band handling reads the
                         PROFILE rather than the module-level LATE_TARGET_CELL.
                         sizing_analysis_core's loop at :3927 then iterates zero
                         times and needs no edit.

RESOLVERS, NOT MUTATED CONSTANTS. Add
    roles_for_schema(schema)      -> _ROLES for <5, the v18 tuple for 5
    splits_for_schema(schema)     -> SPLITS for <5, ("all",) for 5
    allocation_for_schema(schema) -> SPLIT_ALLOC_V2 for <5, the v18 map for 5
mirroring the existing precedent at :246-252
    (`return PROFILE_RUN_KINDS_V3 if int(schema) >= 3 else PROFILE_RUN_KINDS`).
`_ROLES`, `SPLIT_ALLOC_V2`, `SPLITS`, `PROFILE_RUN_KINDS` and every other
historical constant keep their current values and types.
```

**The one-split implementation surface is larger than revision 8 admitted.** Three `AllocationProfile` properties were named; the two-split assumption is in fact spread across the live selection path:

| site | hardcoding |
|---|---|
| `:265-296` | `corpus_size`, `split_totals`, `quota_by_phase` use the `"tuning"`/`"frozen_check"` literals |
| `:381` | allocation parsing, the per-cell `set(counts) != set(SPLITS)` check |
| `:397` | the `corpus_size` cross-check, `sum(a["tuning"] + a["frozen_check"])` |
| `:818` | the two-way assignment: `u_t, u_f = realizable("tuning"), realizable("frozen_check")`, `need[c]["tuning"]`, `need[c]["frozen_check"]` |
| `:1240` | fill, side balance and witness loops iterate the module-level `SPLITS` |

So `sample_v2_rows` **cannot** run unmodified, as revision 8 promised. The scientific choice of one split stands; the frozen route is:

```text
schemas 1-4    retain the CURRENT two-way assignment algorithm, untouched
schema 5       a deterministic ONE-SPLIT assignment mapping every retained
               game to "all"

sample_v2_rows's `stats` (its second return value) exposes
`stats["split_assignment"]` over EVERY retained game, not only the games
selection drew from. Its shape is a **deterministically sorted list of
records**, never a map keyed by game index:

```text
[{"game_idx": 1, "split": "all"}, {"game_idx": 7, "split": "all"}, ...]
sorted by game_idx ascending
```

A `{game_idx: split}` mapping would serialize integer identities as JSON object
keys (`{"1": "all"}`) and silently change their type on reload. A sorted record
list round-trips with `game_idx` still an int, and it is canonically ordered
without relying on dict iteration order. That map is the frozen contract's
observable: without it, a test can only inspect selected rows, which a two-way
assigner could satisfy while mis-assigning the rest of the reservoir.

**Emitted for schema 5 ONLY.** Revision 14 said "populated for every schema;
byte-identical content for schemas 1-4", which is self-contradictory: `stats`
is serialized into selector artifacts, so adding a key that does not exist today
necessarily changes legacy bytes. Schemas 1-4 must not carry it at all. Tests
assert both halves: the key is ABSENT under schemas 1-4, and their full golden
bytes are unchanged.
every parser, capacity-demand, fill, verification, side-balance and witness
loop reads `alloc.splits` instead of the module-level SPLITS
late-band handling reads the profile, never LATE_TARGET_CELL
```

Mutation tests pin **both** directions: the one-split path must break if it silently falls back to two-way assignment, and schemas 1–4 must remain byte-identical if the shared loops are touched.

**`alloc.splits` must exist, and schema 5's allocation must be an acceptance rule.** Revision 9 used `alloc.splits` without defining it, and gave `allocation_for_schema(5)` as a producer default only — so a schema-5 config could move a quota between cells, keep the total at 40, and still be accepted as authoritative. Frozen:

```text
AllocationProfile.splits        a SCHEMA-DERIVED property returning
                                splits_for_schema(self.schema_version).
                                Not a stored field, so it cannot disagree with
                                the schema it claims.

late-band source                a schema-derived CELL IDENTITY on the profile:

    AllocationProfile.band_floor_cell
        ("target", "late")  for schemas 1-4   (the current LATE_TARGET_CELL)
        None                for schema 5
    and: band_minima_total and band_minima_per_split MUST be empty whenever
    band_floor_cell is None; a non-empty minima map with no cell to constrain
    is a parse error, not an ignored field.

    Revision 10 said the floors come from "the PROFILE's own optional
    cell/geometry, never the module-level LATE_TARGET_CELL" -- which is not
    implementable as stated. `band_minima_total` and `band_minima_per_split`
    (:273-274) hold only COUNTS; the cell they constrain is supplied by
    `LATE_TARGET_CELL: Tuple[str, str] = ("target", "late")` (:616), consumed at
    :404, :432, :703, :1140 and :1258. The profile has no cell identity to read.
    `band_floor_cell` adds exactly that identity, and every floor-accounting and
    reporting site routes through it instead of the module constant. The
    constant itself keeps its value and its :617-619 assertions.

    Tests pin the invalid combinations -- schema 5 with non-empty minima, a
    schema 1-4 profile whose band_floor_cell is absent from its allocation --
    and pin legacy behaviour: for schemas 1-4 the property must equal
    LATE_TARGET_CELL and every floor result must be byte-identical.

schema-5 ACCEPTANCE (not merely the default)
    parsed phase_allocation MUST EQUAL allocation_for_schema(5) exactly.
    Refuse on: a missing cell, an extra cell, any altered count, a split name
    other than "all", an unknown role, or a role/phase pair outside the frozen
    table. Cell ORDER is normalized before comparison so a reordered config is
    accepted only if it is otherwise identical -- but the normalization itself
    is recorded, so reordering cannot silently change `cell_order` and hence
    selection order.
    A correct grand total is NOT sufficient: equality is per cell.

    RUN KIND AND LABEL -- enforced by a SCHEMA-5-LOCAL RULE, not by delegating
    to eval_runner.

        schema-5 run_kind             exactly "v18_preflight_sizing"
        profile_run_kinds_for(5)      ("v18_preflight_sizing",)
        required label in the config  exactly True

    Revision 14 routed schema 5 through the existing
    `interpretation_forbidden(run_kind)` check (:353-367). That cannot work:
    `eval_runner.interpretation_forbidden` **raises** `ValueError` on an unknown
    kind by design (":the label boundary must not invent a status for a name it
    does not recognise", eval_runner.py:56-66), `"v18_preflight_sizing"` is not
    in `KNOWN_RUN_KINDS`, and Task 8 does not modify `eval_runner.py`. So
    `base5()` would still have failed after every planned edit.

    Registering the name globally is also the wrong fix: it would widen the
    match runner's accepted label surface for a kind that never runs a match.
    Instead the schema-5 branch asserts the exact pair locally and never calls
    `interpretation_forbidden`. Schemas 3-4 keep the delegating path unchanged.

    A v18 profile therefore cannot claim "production", and cannot be relabelled
    interpretable, without failing the parser.

    REQUIRED KEYS. `parse_allocation_profile` demands
    ("phase_allocation", "late_floors", "late_target_band_minima",
     "max_per_game", "min_ply_gap", "side_tol", "corpus_size")
    plus the schema >= 3 label. A schema-5 profile supplies all of them:
    `late_floors` and `late_target_band_minima` are EMPTY (band_floor_cell is
    None), and `corpus_size` is 40 -- cross-checked at :397 against the
    allocation total, which is itself one of the hardcoded-splits sites the
    dispatch must generalize.

    Every refusal above gets a PARAMETERIZED parser test -- one case per
    rejection reason (missing cell, extra cell, altered count, wrong split
    name, unknown role, out-of-table role/phase pair, non-empty band minima
    under schema 5), each asserting the specific error. Revision 10 stated
    these only in prose.

fingerprinting                  the schema-5 fingerprint additionally records
                                the effective ASSIGNMENT STRATEGY
                                ("one_split" vs "two_way"), so an artifact
                                cannot be reinterpreted under the other path.
                                Schemas 1-4 fingerprints and bytes are
                                unchanged -- pinned by the Step 0 fixture.
```

This matters because allocation is already config-authoritative rather than constant-driven: v17's live config carries a five-cell `phase_allocation` (`target|late`, `control|{opening,early_mid,midgame,late}`), not the eight-cell `SPLIT_ALLOC_V2`. Schema 5 follows the same pattern, so the frozen 40-row table is the thing the parser *enforces*, not a suggestion it starts from.

- Modify `fpu_dev_corpus_v2.py` per the contract above; v18 supplies its vocabulary and allocation through the same config path v2 already parses.
- **Generalize the existing `post_screen_qualification_report` and `sample_v2_rows` to read `alloc.splits`** and dispatch the assignment algorithm on schema. **Do not implement a second exact selector inside `v18_selector_sizing.py`** — one selector, as the design requires.
- Make **late-target-band reporting optional and config-driven**; it is v2 geometry and must not be required under the v18 schema.
- **Preserve the current v2 path byte-for-byte** as the default: calling with no schema argument must produce identical output.
- Reject v18 roles under v16/v17 schemas and run identities.
- Add **real v17 producer byte-regression tests** against a **pre-edit identity basis captured before `fpu_dev_corpus_v2.py` is touched** (Step 0 below). "Feed genuine v17 inputs and compare" is insufficient on its own: if the baseline is generated after the edit, the test compares the change to itself.
- Reuse unchanged: `_binomial_lower_bound`, the whole-game sampling unit, the `n_trials = 1` degenerate full-reservoir branch, and the zero-yield retention in `all_game_ids`.

`v18_selector_sizing.py` therefore contains the v18 role predicates, the tier ladder driver and the record emitter — and no selection algorithm.

`tests/test_fpu_dev_corpus_v2_v18_hook.py` contains, as runnable entries rather than prose:

```python
import copy
import pytest
from scripts.GPU.alphazero import fpu_dev_corpus_v2 as V


def base5():
    """A fresh, minimal, VALID schema-5 profile dict on every call.

    A factory, not a module constant: revision 12 wrote `BASE5 = {...}`, which
    is a set containing Ellipsis, and then mutated it with a shallow `dict()`
    copy -- so nested edits would have leaked into later parameterized cases.
    """
    return {
        "config_schema_version": 5,
        # Frozen run kind + label. The parser requires the label to EQUAL
        # interpretation_forbidden(run_kind) (:353-367), so "production" would
        # both breach the v18 run-identity boundary and fail validation.
        "run_kind": "v18_preflight_sizing",
        "scientific_interpretation_forbidden": True,
        "phase_allocation": {f"{role}|{phase}": dict(counts)
                             for (role, phase), counts
                             in V.allocation_for_schema(5).items()},
        # Required keys revision 12 omitted, so every positive case refused
        # before reaching any schema-5 logic.
        "late_floors": {},
        "late_target_band_minima": {},
        "corpus_size": 40,
        "max_per_game": 1, "min_ply_gap": 12, "side_tol": 0,
    }


def test_base5_is_actually_accepted_by_the_parser():
    """Guard the guard: if base5() drifts out of validity again, every positive
    test below would fail for the wrong reason."""
    assert V.parse_allocation_profile(base5(), source="test") is not None


@pytest.mark.parametrize("kind", ["production", "tooling_smoke"])
def test_schema5_refuses_non_v18_run_kinds(kind):
    cfg = copy.deepcopy(base5())
    cfg["run_kind"] = kind
    with pytest.raises(ValueError, match="run_kind"):
        V.parse_allocation_profile(cfg, source="test")


def test_schema5_refuses_an_interpretable_label():
    cfg = copy.deepcopy(base5())
    cfg["scientific_interpretation_forbidden"] = False
    with pytest.raises(ValueError, match="contradicts run_kind"):
        V.parse_allocation_profile(cfg, source="test")


@pytest.mark.parametrize("mutate, err", [
    (lambda c: c["phase_allocation"].pop("target|late"),        "missing cell"),
    (lambda c: c["phase_allocation"]["target|late"].update({"all": 15}),
                                                                "altered count"),
    (lambda c: (c["phase_allocation"]["target|late"].clear(),
                c["phase_allocation"]["target|late"].update({"tuning": 16})),
                                                                "split name"),
    (lambda c: c["phase_allocation"].update({"bogus|late": {"all": 1}}),
                                                                "unknown role"),
    # A KNOWN role in a phase the frozen table does not allocate. Distinct from
    # "unknown role" above, and no longer duplicated with an "extra cell" case:
    # revision 12 used the same target|opening mutation for two different
    # expected errors, so at most one of them could ever have passed.
    (lambda c: c["phase_allocation"].update({"target|opening": {"all": 4}}),
                                                       "out-of-table role/phase"),
    (lambda c: c.update({"late_target_band_minima": {"all": {"b400_plus": 1}}}),
                                                       "band minima"),
])
def test_schema5_parser_refusals(mutate, err):
    cfg = copy.deepcopy(base5())
    mutate(cfg)
    with pytest.raises(ValueError, match=err):
        V.parse_allocation_profile(cfg, source="test")


def test_schema5_exact_allocation_accepted():
    prof = V.parse_allocation_profile(base5(), source="test")
    assert prof.allocation == V.allocation_for_schema(5)
    assert prof.corpus_size == 40


def test_correct_grand_total_with_shifted_quotas_is_still_refused():
    cfg = copy.deepcopy(base5())
    cfg["phase_allocation"]["target|late"]["all"] = 15
    cfg["phase_allocation"]["representative|late"]["all"] = 5   # total still 40
    with pytest.raises(ValueError):
        V.parse_allocation_profile(cfg, source="test")


def test_schema5_splits_and_band_floor_cell():
    prof = V.parse_allocation_profile(base5(), source="test")
    assert prof.splits == ("all",)
    assert prof.band_floor_cell is None


def test_schema1_legacy_splits_and_band_floor_cell_unchanged():
    """Schema 1 is the only schema the module CONSTANTS canonically describe."""
    prof = V.AllocationProfile.legacy()
    assert prof.splits == V.SPLITS
    assert prof.band_floor_cell == V.LATE_TARGET_CELL


@pytest.mark.parametrize("name", REAL_PROFILE_FIXTURES)   # schemas 2, 3, 4
def test_real_legacy_profiles_are_unchanged(name):
    """Schemas 2-4 are CONFIG-AUTHORITATIVE, so they are tested from tracked
    real profiles -- never manufactured from SPLIT_ALLOC_V2.

    A synthesized schema-4 profile would be fiction: the live v17 config's
    phase_allocation has five cells (target|late plus control| each phase), not
    the eight of SPLIT_ALLOC_V2, so a synthetic fingerprint could stay green
    while real v17 behaviour drifted.
    """
    prof = V.parse_allocation_profile(_real_profile(name), source=name)
    assert prof.splits == V.SPLITS
    assert prof.band_floor_cell == V.LATE_TARGET_CELL
    assert prof.fingerprint() == REAL_PROFILE_FINGERPRINTS[name]


def test_schema5_fingerprint_records_the_assignment_strategy():
    fp = V.parse_allocation_profile(base5(), source="test").fingerprint()
    assert fp["assignment_strategy"] == "one_split"


def test_schema1_legacy_fingerprint_is_unchanged():
    assert V.AllocationProfile.legacy().fingerprint() == SCHEMA1_LEGACY_FINGERPRINT


@pytest.mark.parametrize("name", REAL_PROFILE_FIXTURES)   # schemas 2, 3, 4
def test_split_assignment_is_absent_under_legacy_schemas(name):
    """The key is schema-5 only: `stats` is serialized, so emitting it for
    legacy schemas would change their bytes."""
    _rows, stats = V.sample_v2_rows(
        _real_rows(name), seed=1,
        alloc=V.parse_allocation_profile(_real_profile(name), source=name))
    assert "split_assignment" not in stats


@pytest.mark.parametrize("name", REAL_PROFILE_FIXTURES)
def test_legacy_selector_bytes_are_unchanged(name):
    from scripts.GPU.alphazero.fpu_dev_reservoir_protocol import canonical_json_bytes
    rows, stats = V.sample_v2_rows(
        _real_rows(name), seed=1,
        alloc=V.parse_allocation_profile(_real_profile(name), source=name))
    assert canonical_json_bytes({"rows": rows, "stats": stats}) == \
        (FIXTURE_DIR / f"{name}.selector_output.json").read_bytes()


def test_one_split_assignment_covers_every_retained_game():
    """The frozen contract is about the ASSIGNMENT MAP, not the selected rows.

    Revision 13 checked only the 40 selected rows, which says nothing about the
    retained games that selection did not draw from -- a two-way assigner could
    place half the reservoir in "frozen_check" and still emit 40 rows all
    labelled "all".
    """
    rows, stats = V.sample_v2_rows(
        FIXTURE_ROWS_V18, seed=1,
        alloc=V.parse_allocation_profile(base5(), source="test"))
    retained = {r["game_idx"] for r in FIXTURE_ROWS_V18}
    assignment = stats["split_assignment"]          # sorted list of records
    assert [a["game_idx"] for a in assignment] == sorted(retained)  # total + ordered
    assert {a["split"] for a in assignment} == {"all"}              # single split


def test_split_assignment_survives_a_json_round_trip_with_int_game_idx():
    """A {game_idx: split} map would come back with STRING keys."""
    import json
    _rows, stats = V.sample_v2_rows(
        FIXTURE_ROWS_V18, seed=1,
        alloc=V.parse_allocation_profile(base5(), source="test"))
    reloaded = json.loads(json.dumps(stats["split_assignment"]))
    assert reloaded == stats["split_assignment"]
    assert all(isinstance(a["game_idx"], int) for a in reloaded)


def test_one_split_selected_rows_are_all_in_the_single_split():
    # sample_v2_rows returns (rows, stats) -- Tuple[List[dict], dict] at :1392-1394.
    rows, _stats = V.sample_v2_rows(
        FIXTURE_ROWS_V18, seed=1,
        alloc=V.parse_allocation_profile(base5(), source="test"))
    assert {r["split"] for r in rows} == {"all"}
    assert len(rows) == 40


def test_default_call_path_is_byte_identical_to_the_pre_edit_basis(): ...
def test_v18_role_under_a_v17_schema_raises(): ...
```

`SCHEMA1_LEGACY_FINGERPRINT`, `REAL_PROFILE_FIXTURES`, `REAL_PROFILE_FINGERPRINTS` and `FIXTURE_ROWS_V18` come from the tracked selector-core fixture written in Step 0; `_real_profile(name)` loads a tracked real schema-2/3/4 profile from it.

**`legacy_profile_for(schema)` is NOT added.** Revision 12 referenced it and revision 13 proposed adding it to production as a constructor over `SPLIT_ALLOC_V2` for schemas 1-4. That would manufacture history: the module constants canonically describe only the **schema-1** legacy path, while schemas 2-4 are config-authoritative — and the live v17 schema-4 config carries a five-cell `phase_allocation`, not the eight of `SPLIT_ALLOC_V2`. A synthetic schema-4 fingerprint could stay green while real v17 behaviour drifted, which is the opposite of what a regression test is for. Instead: keep the existing `AllocationProfile.legacy()` for schema 1, and test schemas 2-4 from **tracked real profiles** captured in Step 0. No new production test-support constructor.

**Interfaces:**
- Consumes: `v18_preflight_criteria.{EXPOSURE_CUTOFF_RULE, FLIP_CONTROL_EXPOSURE, SIGN_DOMINANCE, SIZING, ...}`, the Task 7 `census_positions.csv`, the frozen Task 4 universe record, **the Task 4b matched-cohort artifact** (`EXPOSURE_CUTOFF` is derived from those 30 rows, so Task 8 cannot compute role predicates without it), `fpu_dev_corpus_v2.{_binomial_lower_bound, sizing_analysis_core}` (import; do not fork), `canonical_json_bytes`. Every input is authenticated by SHA-1 before use.

**Data path, which revision 2 omitted.** Sizing needs a whole-universe census joined back to reservoir prefixes, not the final cohort:

```text
sizing universe  = the frozen Task 4 source universe (800 games)
census           = Task 7's per-game measured positions over that universe
join key         = replay content SHA-1 (game) + position_ply
telemetry        = shipped residual exposure per census position, from Task 7
tier ladder      = 200, 300, 400, 500, 600, 700 probabilistic; 800 degenerate
trials           = 299 per probabilistic tier, random subsets under seed
                   20260729; tier 800 is ONE trial, not 299
success          = an EXACT-SELECTOR witness filling the full four-role
                   geometry, never a capacity bound
pass rule        = _binomial_lower_bound(k, 299, alpha=0.05) >= 0.99, imported
                   from fpu_dev_corpus_v2:3876 (exact one-sided Clopper-Pearson)
witness          = one successful trial at the passing tier, named by its trial
                   index under the frozen seed -- NOT the content-SHA prefix
```

If tier 700 does not qualify, that is a **completed sizing failure** and Task 9
returns `PREFLIGHT_FAIL` — not a licence to enlarge the universe after seeing
the shortfall, and not something 299 repetitions of the identical 800-game set
can rescue.
- Produces: `exposure_cutoff(control_rows) -> float`; `role_predicates(cutoff) -> dict`; `classify_rows(rows, predicates) -> dict`; `sizing_ladder(rows, predicates, tiers, trials, seed) -> list[dict]`; `operating_characteristics(...) -> dict`; `emit_sizing_record(path, ...) -> str`.

Role predicates, from spec §9.2.1, all shipped-only:

```text
target             : exposure >= EXPOSURE_CUTOFF
                     and sign_dominance >= 0.80
                     and near-even and minimum-eligible-leaf rules
identity_witness   : max|eligible depth-2 residual| <= 0.50
flip_control       : v18_preflight_criteria.FLIP_CONTROL_EXPOSURE, IMPORTED
                     -- count(|residual| > 1.25) >= 3 AND clipped amount at
                     1.25 >= 0.50. The operator is AND. Revision 3 restated it
                     here as "and/or", weakening the frozen predicate; Task 8
                     must import the predicate, never restate it.
representative     : phase-balanced, selected independently of exposure
```

- [ ] **Step 0: Capture the pre-edit v17 identity basis — BEFORE any edit**

`fpu_dev_corpus_v2.py` must still be at its pinned content SHA-1 when this runs. The real v17 producer chain and its artifacts:

```text
inputs   logs/eval/fpu_v17_baseline_policy_mass/development/
             fpu_dev_corpus_v2_config.json      the allocation config
             fpu_dev_source_screen.csv          the screened rows
outputs  logs/eval/fpu_v17_baseline_policy_mass/development/
             fpu_dev_corpus_v2_manifest.csv
             fpu_dev_corpus_v2_manifest.csv.meta.json
```

**None of those four files is tracked.** `logs/*` is gitignored (`.gitignore:31`), and `git ls-files` confirms the manifest is untracked — revision 8 called them "committed outputs", which is wrong. A suite test cannot depend on them. Split the regression in two:

**The verbatim config cannot be paired with a reduced screen.** That config binds `source_index_path`, `protocol_path`, `replay_dir`, `match_summary_path` and `screen_out`, plus an `expected_fingerprints` block carrying `protocol_sha1`, `match_summary_sha1`, `replay_data_sha1`, `source_file_sha1s` and both checkpoint identities. Substituting the screen bytes makes the authenticated producer chain refuse, and the reduced set would in any case lack the source index and sidecars that chain requires. Revision 9's fixture is therefore not constructible as described. Of the two contracts available, **(a) is chosen** — a selector-core fixture — with **(b)** as a separate operator check:

**Real profiles for schemas 2, 3 and 4 — named, not assumed.** Revision 14's tests consume `REAL_PROFILE_FIXTURES` for three schemas while Step 0 produced only a schema-4 artifact. The three real configs on disk, with their live SHA-1s at time of writing:

| schema | run_kind | source config | SHA-1 |
|---|---|---|---|
| 2 | `production` | `logs/eval/fpu_v16_policy_mass_v2/production_v2_b400amend_4000g_seed20300000/fpu_dev_corpus_v2_config.json` | `1a26c88573ba1ac2973eb93bfdda982959fbb366` |
| 3 | `tooling_smoke` | `logs/eval/fpu_v17_baseline_policy_mass/smoke_8a/fpu_dev_corpus_v2_config.json` | `a6c6c51a69a6deffa7fa829de53de1d66bf015ea` |
| 4 | `development` | `logs/eval/fpu_v17_baseline_policy_mass/development/fpu_dev_corpus_v2_config.json` | `9d3ef92172770afa4e1421a5643b05f2a1735c67` |

Step 0 copies each profile input into the tracked fixture directory, alongside a reduced row set and the unedited module's `(rows, stats)` output for each, and records the original path and SHA-1 for all three. Re-verify the hashes at capture time rather than trusting the table. `REAL_PROFILE_FIXTURES` is exactly these three names; `_real_profile(name)` and `_real_rows(name)` load them.

**(a) A tracked, portable SELECTOR-CORE fixture — what the suite runs.** Not a full producer regression, and it must not be described as one. It uses a **tracked fixture profile** (a small schema-4 `AllocationProfile` written for the test, not the v17 config) and a reduced set of screen-shaped rows, and exercises exactly two functions: `post_screen_qualification_report` and `sample_v2_rows`. Coverage — corrected from revision 9, which said "all four roles' v2 equivalents" when v17 has **two** roles:

```text
both roles          target, control
both splits         tuning, frozen_check
all four phases     opening, early_mid, midgame, late
every late band     b400_plus, b300_399, b200_299
```

Commit the profile, the rows and the unedited module's outputs under `tests/golden/v18_v2_selector_pre_edit_basis/`. This is the byte-regression the suite enforces on every run, and its scope is the selector core — which is precisely the surface schema 5 modifies.

**(b) An operator-only full-chain reproduction — not part of the suite.** Re-run the real authenticated producer at the unedited HEAD over the ignored artifacts above, byte-compare against the on-disk outputs, and record all four SHA-1s plus the module's own content SHA-1 alongside the fixture. One-time, recorded rather than tested, because the suite cannot depend on gitignored files.

Both must pass before any edit. If (b) does not reproduce, **stop** — the chain is not reproducible and no regression built on it would mean anything.

- [ ] **Step 0b: AUTHORIZED FIXTURE COMMIT — a hard boundary before Step 1**

Revision 15 said "commit the fixture before editing" with no checkpoint, no named record path and no command, so the baseline could have stayed untracked or — worse — been captured after the edit. This is now its own authorization gate.

Artifacts produced by Step 0, all **tracked**:

```text
tests/golden/v18_v2_selector_pre_edit_basis/
    schema2.profile.json  schema2.rows.json  schema2.selector_output.json
    schema3.profile.json  schema3.rows.json  schema3.selector_output.json
    schema4.profile.json  schema4.rows.json  schema4.selector_output.json
    sources.json                     original paths + SHA-1s for all three
    operator_full_chain_reproduction.json    the (b) result
tests/golden/a6400_bundle_fixture/
    source_rows.json  capture.json
```

Present the user with: the three re-verified config SHA-1s, the (b) reproduction outcome, and `shasum -a 1 scripts/GPU/alphazero/fpu_dev_corpus_v2.py` proving the module is still unedited. **Stop for authorization.** Only then:

```bash
git add tests/golden/v18_v2_selector_pre_edit_basis/ tests/golden/a6400_bundle_fixture/
git commit -m "test(v18): pre-edit selector-core basis and A/6400 bundle fixtures"
```

The commit must land while `fpu_dev_corpus_v2.py` still matches its pinned pre-edit hash. Verify that after committing, before Step 1.

- [ ] **Step 1: Write the failing test**

Create `tests/test_v18_selector_sizing.py`, each written out in full:

```python
def test_exposure_cutoff_uses_control_rows_only(): ...
def test_exposure_cutoff_is_nearest_rank_not_interpolated(): ...   # value is an observed datum
def test_cutoff_ties_are_admitted_by_the_predicate(): ...
def test_role_assignment_is_total_and_exclusive(): ...
    # the frozen order target -> representative -> {identity, flip} produces a
    # PARTITION. Revision 5 asserted mutual exclusivity, which the predicates
    # alone do not establish: a row can satisfy both target and flip exposure.
def test_representatives_are_chosen_before_identity_and_flip_exist(): ...
    # revision 7's correction: their candidate set must be conditioned on
    # target status ONLY, never on identity/flip removal
def test_target_flip_overlap_resolves_to_flip_control(): ...
def test_representative_ordering_key_is_residual_independent(): ...
def test_role_assignment_refuses_rather_than_reordering(): ...
def test_identity_witness_predicate_cannot_bind_at_any_grid_cap(): ...
def test_flip_control_predicate_requires_material_exposure_at_1_25(): ...
def test_flip_control_exposed_at_1_25_is_exposed_at_every_stronger_cap(): ...
def test_sizing_ladder_reports_the_smallest_qualifying_tier(): ...
def test_sizing_ladder_is_deterministic_under_a_frozen_seed(): ...
def test_sizing_reports_operating_characteristics_per_count_gate(): ...
def test_infeasible_geometry_fails_rather_than_relaxing(): ...
def test_sizing_record_is_byte_reproducible_and_forbids_interpretation(): ...
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest -p no:cacheprovider tests/test_v18_selector_sizing.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the implementation**

Supply the v18 role vocabulary, split vocabulary and allocation through the frozen **schema-5 dispatch** above — not through callbacks, which revision 7 retired. `post_screen_qualification_report` and `sample_v2_rows` keep **one** implementation each, generalized to read `alloc.splits`; they are not forked and not re-implemented, but they are not unmodified either (spec §5.1: parameterize, do not fork). Import `_binomial_lower_bound` unchanged. Apply the whole-game exact-selector sizing discipline established by the v2 repair: a preregistered resampling seed, a tier ladder, a trial count, an exact success criterion and a next-tier-up margin rule; **PASS is an exact-selector witness, never a capacity bound**. Emit a canonical record with the chosen cutoff, per-role counts, the smallest qualifying tier, the recommended operational reservoir size, per-count-gate operating characteristics, and `scientific_interpretation_forbidden: true`. Infeasible geometry raises rather than relaxing quotas.

- [ ] **Step 4: Run the test to verify it passes, then the full suite**

Expected: PASS, zero failures.

- [ ] **Step 5: Request authorization, then commit**

```bash
git add scripts/GPU/alphazero/v18_selector_sizing.py \
        scripts/GPU/alphazero/fpu_dev_corpus_v2.py \
        tests/test_v18_selector_sizing.py \
        tests/test_fpu_dev_corpus_v2_v18_hook.py \
        tests/golden/v18_v2_selector_pre_edit_basis/
git commit -m "feat(v18): schema-dispatched role vocabulary in the v2 selector, plus v18 sizing"
```

The fixture directory appears here only to catch anything Step 0b did not already commit; if that gate was honoured, `git add` on it is a no-op. If it is *not* a no-op, the baseline was captured late — stop and re-run Step 0 at the pinned pre-edit hash.

```bash
```

---

### Task 9: Preflight verdict evaluator

**Files:**
- Create: `scripts/GPU/alphazero/v18_preflight_verdict.py`
- Test: `tests/test_v18_preflight_verdict.py`

**Interfaces:**
- Consumes: `v18_preflight_criteria`, the Task 7 artifact **and `census_positions.csv`**, the **Task 4b matched-cohort artifact**, the **A-reference bundle** (below), and the Task 8 sizing record. Each is authenticated by SHA-1 before being read.

**A-reference bundle.** Revision 6 said the verdict records both A/6,400 capture SHA-1s but gave `evaluate(...)` no capture input, so the requirement had nowhere to live. Execution step 4 now writes `a6400_reference_bundle.json` carrying:

```text
capture_run_1_path, capture_run_1_sha1
capture_run_2_path, capture_run_2_sha1
byte_identical                 bool -- runs 1 and 2 compared
historical_source_path         logs/eval/v15_budget_check/
                                 a_predrop_base_6400sims.csv/
                                 position_probe_cases.csv
historical_source_sha1         a17d4737c747e2799253bebbc3d0261e0e697114
authentication                 per-case value and top-share result, 30 cases
run_kind, scientific_interpretation_forbidden
```

**The bundle is re-derived, never trusted.** Revision 7 listed the fields and named no producer, so nothing owned the file and the verifier could have believed a written `byte_identical: true`. Frozen:

- **Producer** — `build_a6400_reference_bundle`, owned by **Task 6** where the captures are made. It emits canonically with `artifact_kind`, `schema_version: 1` and an exact key set, and **returns** the SHA-1 of the bytes it wrote; the bundle never claims its own digest.
- **Verifier — two unambiguous entry points.** Revision 11 declared a path-based API while requiring `evaluate` to hash bytes and verify "those exact bytes", which cannot both hold. Frozen:

```text
verify_a6400_bundle_bytes(raw: bytes, *, bundle_path: str) -> dict
    the real verifier. `raw` is what gets verified; `bundle_path` is still
    needed to resolve referenced capture paths and to restrict them to the
    frozen artifact root.

load_verified_a6400_bundle(path) -> dict
    optional convenience: reads once and delegates to the byte verifier.
```

`evaluate` reads the file once, hashes `raw` into `a6400_reference_bundle_sha1`, and calls `verify_a6400_bundle_bytes(raw, bundle_path=path)` — so the bytes hashed and the bytes verified are provably the same object, with no second read in between.

The verifier **recomputes every claim rather than reading it**:
  1. reopen both captures and verify their live SHA-1s against the bundle;
  2. compare the two captures' canonical bytes **itself**, ignoring the stored `byte_identical`;
  3. reopen and hash the historical source, requiring `a17d4737c747e2799253bebbc3d0261e0e697114`;
  4. recompute all 30 per-case value and top-share authentications from the reopened files;
  5. reject missing or extra keys, and reject any path that resolves outside the frozen artifact root.

The five tamper attacks live **only** here, with the verifier. A builder cannot be attacked by editing its own output, so Task 6 carries emission and refusal tests instead.

`evaluate(...)` takes the bundle as an input and refuses on any of those failures. Stage 0 can then bind the accepted reference from the verdict rather than rediscovering an orphan file.
- Produces: `separation(cohort_artifact, a_rows, criteria) -> dict`; `decide_revisit_form(census, criteria) -> str`; `derive_thresholds(census, cohort_artifact, criteria) -> dict`; **`verify_a6400_bundle_bytes(raw: bytes, *, bundle_path: str) -> dict`**; `load_verified_a6400_bundle(path) -> dict` (reads once, delegates); `evaluate(artifact, cohort_artifact, a_reference_bundle_path, sizing_record, criteria) -> dict`.

**Population split, per Task 5.** `separation` and `EXPOSURE_CUTOFF` read the 30-row **matched cohort** — matching is what makes them comparable to A row-for-row. `decide_revisit_form` and `R_min` read the **broad non-A census**, because the cohort's top exposure decile is about three rows.

**Verdict vocabulary.** `evaluate` returns exactly one of:

- `PREFLIGHT_FAIL` — any mechanism criterion failed, **or** sizing ran to completion and could not satisfy the frozen geometry.
- `MECHANISM_PREFLIGHT_PROVISIONAL_PASS` — every mechanism criterion passed and the sizing record is **absent or not yet run**. Not a preflight pass; authorizes nothing.
- `PREFLIGHT_PASS` — mechanism criteria and sizing both passed. Only this may be presented as satisfying spec §2.3.

**Missing sizing and failing sizing are different states.** Revision 2 mapped both to provisional, which meant an infeasible selector could never formally reject v18 — it would sit at "provisional" indefinitely. A completed sizing run that cannot fill the four-role geometry is a real negative result about the mechanism's practicality and rejects.

- [ ] **Step 1: Write the failing test**

Create `tests/test_v18_preflight_verdict.py`, each written out in full:

```python
def test_thresholds_derive_only_from_non_a_rows(): ...
def test_reach_is_measured_on_a_rows_only(): ...
def test_separation_reads_the_matched_cohort_not_the_broad_census(): ...
def test_r_min_reads_the_broad_census_not_the_matched_cohort(): ...
def test_verdict_authenticates_the_matched_cohort_artifact_sha1(): ...
def test_verdict_refuses_a_cohort_artifact_that_is_not_exactly_30_rows(): ...
def test_r_min_failure_reason_matches_the_frozen_string(): ...
    # must be "mechanism_not_predicted_to_act_at_any_cap", the Task 5 constant;
    # revision 3 asserted the obsolete "mechanism_not_predicted_to_act"
def test_separation_uses_the_single_frozen_primary_formula(): ...
def test_descriptive_formulas_cannot_rescue_a_failed_primary(): ...
def test_separation_requires_both_point_estimate_and_bootstrap_lower_bound(): ...
def test_r_min_fails_on_a_nonpositive_pooled_prediction(): ...
def test_r_min_is_not_floored_up_from_a_negative_prediction(): ...
def test_empty_r_band_is_a_failure(): ...
def test_revisit_form_paired_when_prospective_targets_are_dense(): ...
def test_revisit_form_candidate_only_when_sparse(): ...
def test_sign_dominance_failure_rejects(): ...
def test_terminal_fraction_over_bound_rejects(): ...
def test_missing_sizing_record_yields_provisional_not_pass(): ...
def test_completed_but_failing_sizing_yields_PREFLIGHT_FAIL_not_provisional(): ...
def test_full_pass_requires_both_mechanism_and_sizing(): ...
def test_verdict_is_byte_reproducible(): ...
def test_verdict_records_every_input_sha1(): ...
def test_verdict_records_both_a6400_capture_sha1s(): ...
def test_undersized_prospective_target_subset_is_preflight_fail(): ...
    # reason prospective_target_subset_below_floor, floor 16 rows
# The five tamper attacks, listed explicitly. Revision 10 promised five and
# named two. Each must be caught by RECOMPUTATION, not by reading a field.
def test_attack_forged_byte_identical_true_over_differing_captures(): ...
def test_attack_capture_altered_after_hashing(): ...
    # bundle's stored capture_run_1_sha1 no longer matches the live file
def test_attack_fabricated_per_case_authentication_block(): ...
    # every case marked pass in the bundle, but recomputation disagrees
def test_attack_substituted_historical_source_path(): ...
    # a look-alike file whose hash is not a17d4737...
def test_attack_missing_or_extra_keys_rejected(): ...
def test_verdict_records_the_bundle_document_sha1(): ...
    # a6400_reference_bundle_sha1, computed over the bytes actually verified
def test_verdict_hashes_and_verifies_the_same_bytes(): ...
    # no re-read between hashing and verification
```

`test_thresholds_derive_only_from_non_a_rows` must build inputs whose A rows would yield a *different* threshold than the non-A populations, and assert the derived value matches the non-A ones — **the winner's-curse guard, and it must be able to fail.** `test_r_min_is_not_floored_up_from_a_negative_prediction` must assert `PREFLIGHT_FAIL` with the frozen reason `mechanism_not_predicted_to_act_at_any_cap`, never `R_min == 0.01`; assert against `v18_preflight_criteria`'s constant rather than a string literal, so the two cannot drift apart again.

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest -p no:cacheprovider tests/test_v18_preflight_verdict.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the implementation**

Implement exactly the formulas frozen in Task 5. `derive_thresholds` raises when `R_min >= R_max`. `evaluate` assembles the record, applies the verdict vocabulary above, and records every input SHA-1: criteria, universe record, `census_positions.csv`, matched-cohort artifact, sizing record — **and the A-reference bundle document itself**.

**Binding the bundle, not just its contents.** Revision 10 recorded the two capture hashes but never the bundle's own digest, so the accepted *document* was unbound and Stage 0 would have had to reconstruct it. Required:

```text
evaluate(..., a_reference_bundle_path, ...)      takes the PATH, not a dict
  1. raw = open(path, "rb").read()                     -- the ONLY read
  2. a6400_reference_bundle_sha1 = sha1(raw)           -- recorded in the verdict
  3. verify_a6400_bundle_bytes(raw, bundle_path=path)  -- same bytes object, so
     the hashed and verified bytes provably cannot differ
  4. record capture_run_1_sha1 and capture_run_2_sha1 FROM THE VERIFIED RESULT,
     never from the raw file
```

Stage 0 then binds the accepted document by `a6400_reference_bundle_sha1`.

- [ ] **Step 4: Run the test to verify it passes, then the full suite**

Expected: PASS, zero failures. Confirm the Task 0 invariant: `shasum -a 1 scripts/GPU/alphazero/mcts.py` equals the pinned content SHA-1, and `git diff --stat <pinned branch point>..HEAD -- scripts/GPU/alphazero/mcts.py` is empty. **Not** a comparison against `main`.

- [ ] **Step 5: Request authorization, then commit**

```bash
git add scripts/GPU/alphazero/v18_preflight_verdict.py tests/test_v18_preflight_verdict.py
git commit -m "feat(v18): preflight verdict evaluator with winner's-curse guard and provisional-pass vocabulary"
```

---

## Execution Phase — after all eleven tasks are committed

Revision 1 placed the real measurement inside Task 7, before Task 7 was committed and before Tasks 8–9 existed. That was wrong twice: the CLI itself requires a clean worktree, so the run could not happen there; and writing the evaluator after seeing measurements leaves room for the implementation to be shaped by the result. The corrected order:

- [ ] **Step 1: Reach a clean, final tooling HEAD**

All eleven tasks committed (0, 1, 2, 3, 4, 4b, 5, 6, 7, 8, 9). `git status --porcelain` empty. The Task 0 invariant holds: `mcts.py` content SHA-1 matches the pin, and its diff against the **pinned branch point** is empty. Full suite green. Record the HEAD SHA — every artifact below binds to it.

- [ ] **Step 2: Emit and independently authenticate the frozen criteria**

```bash
.venv/bin/python -c "from scripts.GPU.alphazero import v18_preflight_criteria as C; \
print(C.emit_frozen_criteria('logs/eval/v18_depth2_provisional_backup/frozen_preflight_criteria.json'))"
```

The record now carries a clean worktree and the final HEAD. **Authenticate independently:** re-derive the criteria dict from the committed module in a fresh interpreter, emit to a temporary path, and byte-compare against the bound file. Do not trust a SHA stored beside the artifact — an edit to both file and neighbour would pass.

- [ ] **Step 3: Freeze the source universe at this HEAD**

Run `freeze_source_universe` against the source the user chose at Task 4 Step 5, binding the universe SHA-1, the exclusion report, the summary/JSONL/sidecar/checkpoint hashes, and the exact 800-game content-SHA-ordered membership. Then run the derived-seed disjointness audit and record its intersection size.

- [ ] **Step 4: Run the shipped-only measurements (single authorized GPU stage)**

This stage contains **five** GPU runs. Revision 1 claimed one; revision 2 claimed four but still had no measurement for the sizing census:

1. v17 400-sim default regression capture (`--mode v17_prechange_abcd`), byte-compared against `logs/eval/fpu_v17_baseline_policy_mass/prechange_abcd_selected_moves.json`.
2. A/6,400 capture (`--mode v18_preflight_a6400`), run #1, authenticated on exact case identity plus per-case value **and** top share against `a17d4737c747e2799253bebbc3d0261e0e697114`.
3. A/6,400 capture, run #2, byte-compared against run #1. **Record both captures' SHA-1s in the final preflight bundle and in the Task 9 verdict**, so Stage 0 can bind the accepted 6,400 reference into the future protocol instead of rediscovering an orphan file on disk.
4. Residual preflight over `selected_a` — 30 positions at 400 sims.
5. Residual preflight over the **per-game census** of the frozen universe — the large run, and the one that feeds both the Task 4b matcher and Task 8 sizing.

Run 5 dominates the cost and its size is set by the census-per-game factor, not by the cohort size. Measure throughput on a handful of census positions and extrapolate from the **measured rate**, then order the runs cheapest-first so a failure surfaces before the expensive one starts.

Task 4b's matcher runs after run 5, on CPU, before Tasks 8 and 9.

Present measured throughput on a handful of rows first, then a runtime estimate from **measured rate, never from scaling a smoke** — v17's estimates were wrong by 3× one way and 10× the other. Long runs use `nohup` plus `disown`; `setsid` does not exist on macOS. **No source edit and no commit anywhere inside this stage.**

- [ ] **Step 5: Run the already-committed evaluators without modifying them**

Run Task 8's sizing and Task 9's `evaluate` against the artifacts. If either needs a code change, that change invalidates the run: revert, fix, re-commit, and re-run every measurement from step 4.

- [ ] **Step 6: Present the verdict**

Report `PREFLIGHT_FAIL`, `MECHANISM_PREFLIGHT_PROVISIONAL_PASS` or `PREFLIGHT_PASS`, with the derived `R_min` / `R_max` / efficiency floor / exposure cutoff / revisit form. A fail ends v18 with no `mcts.py` edit. A pass authorizes only the **next** conversation: the frozen design revision, and then Stage 0.

## What this plan deliberately does not build

- **No `mcts.py` field, call site, or batched refusal** (spec §4.1/§4.3) — Stage 0, behind preflight PASS.
- **No corpus, reservoir, or generation** (spec §9 onward).
- **No positive-cap search anywhere.**

## Self-review

**Spec coverage.** §2.2.1 crossover → Task 3. §2.2.2 A/6,400 capture → Task 6 plus Execution step 4. §2.2.3 threshold provenance → Task 4's pool, Task 7's `population` tagging, Task 9's winner's-curse test. §2.3 freeze blockers → Tasks 5 and 8. §3.1/§3.2 → Tasks 1–2. §4.2 one-implementation → Task 1, imported everywhere. §4.4 post-hoc telemetry → Task 2. §9.2.1 exposure statistic and roles → Tasks 5 and 8. §10.1.1 → Task 2. Selector sizing, omitted in revision 1, is now Task 8. No spec section in preflight scope lacks a task.

**Placeholder scan.** Tasks 5, 7, 8 and 9 list test *names* with explicit required assertions rather than full bodies; each step requires them written out in full and names the must-be-able-to-fail cases. Tasks 1–3 and 6 carry complete runnable test code. No "TBD", no "similar to Task N".

**Type consistency.** `provisional_depth2_backup_value` returns `ProvisionalBackup` in Tasks 1–3. `walk(root, caps)` keys `per_cap` by `str(cap)` in Task 2, read that way in Tasks 7–9. `CAP_GRID` defined once in Task 1. `exposed_positive_backup_mass` returns `(numerator, denominator)` in Task 2 and is pooled, not averaged, in Task 9. `MIN_LOST_REPLIES` is a dict in Tasks 5 and 9. `mcts_config(mcts_sims)` matches its test in Task 6. The A example is `+0.087 / 0.880 / 0.413` in Tasks 1, 2 and 3.

## Revision 15 change log

Revision 14 was reviewed: four contract blockers, all incorporated.

72. **The new run kind was unknown to the policy it called.**
    `eval_runner.interpretation_forbidden` raises `ValueError` on an unknown
    kind by design (`eval_runner.py:56-66`), `"v18_preflight_sizing"` is not in
    `KNOWN_RUN_KINDS`, and Task 8 does not modify `eval_runner.py` — so
    `base5()` would still have failed after every planned edit. Schema 5 now
    enforces the exact run-kind/`True`-label pair **locally** and never calls
    `interpretation_forbidden`; schemas 3-4 keep the delegating path. Registering
    the name globally is rejected as widening the match runner's accepted label
    surface for a kind that never runs a match.
73. **`split_assignment` contradicted legacy byte identity.** `stats` is
    serialized into selector artifacts, so a new key cannot be added for schemas
    1-4 byte-identically. It is now emitted for **schema 5 only**, with tests
    asserting its absence under legacy schemas and their full golden bytes
    unchanged.
74. **Step 0 did not create the schema-2/3 fixtures the tests consume.** The
    three real configs are now named with their live SHA-1s — schema 2
    `production_v2_b400amend_.../fpu_dev_corpus_v2_config.json`
    `1a26c885…`; schema 3 `smoke_8a/...` `a6c6c51a…`; schema 4
    `development/...` `9d3ef921…` — and Step 0 copies each into the tracked
    fixture with its original path and hash recorded, re-verified at capture
    time.
75. **The loader test did not prove authentication precedes reading.** It
    discarded the operation names, so a read-then-hash loader passed. It now
    asserts the exact sequence `[("hash", frozen), ("read", frozen)]`, and the
    mismatch test asserts **zero** calls to `_read_source_rows`.

Correction: revision 14 was 3,352 lines, not 3,354 as reported.

## Revision 16 change log

Revision 15 was reviewed: three issues, all incorporated.

76. **The frozen source was authenticated and parsed from different reads.**
    `file_sha1(path)` read once and `_read_source_rows(path)` read again, so a
    file changed between them would authenticate one byte sequence and parse
    another -- hash-before-parse ordering cannot catch that. The loader now
    reads **once** into `raw`, hashes that object via `sha1_bytes(raw)`, and
    parses that same object via `_parse_source_rows(raw)`. The test asserts the
    hashed and parsed arguments are the same object (`is raw`), not merely equal.
77. **Step 0 had no executable commit boundary.** Added **Step 0b**, an
    authorization gate that names every tracked fixture path including
    `operator_full_chain_reproduction.json`, requires the three re-verified
    config SHA-1s and proof that `fpu_dev_corpus_v2.py` is still at its pre-edit
    hash, and carries the commit command. Task 8's final `git add` now also
    lists the fixture directory as a late-capture tripwire: if it is not a
    no-op, the baseline was taken after the edit and Step 0 must be re-run.
78. **`split_assignment` used integer JSON object keys.** Now a
    deterministically sorted list of `{"game_idx": int, "split": str}` records,
    with a round-trip test asserting `game_idx` survives as an `int` -- a map
    would have reloaded as `{"1": "all"}`.

## Revision 17 change log

Revision 16 was found defective during Task 2 execution: one unsatisfiable test
and one unspecified representation. Both incorporated. This is the first
revision written after the plan was committed (`a324e55`), so the amendment is
recorded here rather than folded in silently.

79. **A Task 2 test could not pass under any implementation.**
    `test_explored_replies_exclude_terminal_and_ineligible` asserted
    `set(W.explored_replies(d1a)) == {leaf_hi, leaf_lo}`. `MCTSNode` is a plain
    `@dataclass` (`mcts.py:253`), so `eq=True` sets `__hash__ = None` and the
    type is unhashable — the set **literal** on the right-hand side raises
    `TypeError` before the walker is consulted, and no return value can rescue
    it. Repairing it in `mcts.py` is forbidden by the Global Constraints and
    would be a real behaviour change, so the repair is in the test: compare
    `{id(n) for n in got}` against `{id(leaf_hi), id(leaf_lo)}`, which imposes
    neither hashability nor ordering on the declared `list[MCTSNode]` return
    type. Verified: the walker was already correct, returning exactly those two
    nodes by identity.
80. **`walk`'s undefined-statistic representation was unspecified.** The
    standalone functions raise `ValueError` on an empty denominator, but
    `test_walk_emits_every_documented_key_per_cap` passes `caps=(1.25, 0.50)`
    and cap 1.25 clips nothing on the fixture tree — so the plan demanded a
    record whose contents it never defined. Now pinned: `walk` records `null`,
    never a fabricated `0.0`, and never aborts a multi-cap walk; the standalone
    contracts are unchanged. Two assertions at cap 1.25 pin it
    (`would_clip_count == 0`, `revisit_to_depth3_rate is None`), so a later
    refactor to `0.0` fails rather than silently reading as "clipped leaves were
    never revisited".

## Revision 18 change log

Revision 17 was reviewed before Task 3 began: one load-bearing evidence-boundary
defect, incorporated. Task 3 was **not** implemented against revision 17.

81. **`assert_synchronous_tree` proved the simulation budget, not synchronous
    provenance.** It checked only `root.visit_count == expected_sims`. Both
    search entry points back up exactly one path per simulation, so a
    batched-waiter tree satisfies that equality identically — the assertion
    named in the module docstring as the prerequisite for exactness could never
    detect the one condition that breaks it. The failure is silent and total:
    `search_from_root` "backs up ALL waiters with the returned value"
    (`mcts.py:595-606`), so a leaf with `k` waiters holds `k * nn_value` in
    `value_sum` while `counterfactual_child_q` substitutes one, leaving the
    crossover wrong by `(k-1) * (backup_value - nn_value)` with nothing to
    signal it. This is evidence-boundary work, not scaffolding: the substitution
    is exact **only** on the synchronous path (`mcts.py:528-535`), and the whole
    Task 3 → Task 5 → Task 9 chain inherits whatever exactness this assertion
    fails to establish. Four changes:
    - `assert_synchronous_tree(root, expected_sims, *, search_execution_mode)`
      now checks the exact count **and** `search_execution_mode ==
      "synchronous"`. The argument is required and keyword-only with **no
      default** — a default of the safe value would reinstate the hole for every
      caller that forgot it.
    - A negative test supplies a **matching-count** tree tagged
      `"batched_waiter"` and requires refusal, so the test can only pass on an
      implementation that reads the mode. A companion test pins the missing
      argument to `TypeError` and unknown modes to `ValueError`.
    - Task 7 gains a routing test asserting the measurement calls
      `search_with_root` once per position and `search_from_root` **exactly
      zero** times. That dynamic observation is the provenance evidence; a
      source-text check for `search_from_root` is a supplemental tripwire only,
      since source text cannot establish which path executed. The mode is an
      input `v18_crossover` cannot verify; the route is what makes it true, so
      the route is what is tested.
    - The artifact's `search_execution_mode` label is derived from a single
      `SEARCH_EXECUTION_MODE` constant bound to that fixed call site — never a
      CLI flag, never a caller-supplied `meta` key (`build_row` raises on one),
      never an independent literal at the emission site. A label that can be
      asserted by its caller is not evidence. On the valid path this is proven
      by spying on `assert_synchronous_tree`: every call must receive the module
      constant, and the artifact field must equal that same constant. Under a
      **mutated** constant the measurement must raise `ValueError` and write no
      artifact — asserting instead that the label "moves with" the mutation
      would be self-defeating, because a successful emission under a
      non-synchronous mode could only mean the assertion was bypassed. An
      invalid route label yields no artifact, never a valid artifact carrying an
      invalid label.

## Revision 19 change log

Revision 18 was implemented in Task 3: ten of eleven tests passed against a
faithful implementation, and the eleventh was found unsatisfiable. Fixture-only;
no interface, requirement or threshold moved.

82. **The signed-reduction test could not express the effect it guarded.**
    `test_reduction_may_be_negative_for_a_negative_residual_population` asserted
    `predicted_reply_reduction < 0.0` on a fixture whose unvisited priors were
    `{12: 0.3, 13: 0.2}`. The clip behaves exactly as documented — the negative
    residual raises the leaf's counterfactual value and lowers its parent-side
    score from `1.076532` to `1.004309` — but both unvisited moves already
    outscored the shipped best (`2.012461`, `1.341641`), so both counted in both
    arms and the reduction was `0.0`. The count can only move for an unvisited
    move scoring inside the band, which for `c_puct * prior * sqrt(20)` means
    `prior in (0.149714, 0.160480]`; neither `0.3` nor `0.2` is close. No
    implementation faithful to `_select_child` (`mcts.py:1062`, `1091-1114`)
    could pass it, and the assertion `0.0 < 0.0` was unreachable by
    construction. Repaired to `{11: 0.5, 12: 0.345, 13: 0.155}` — priors still
    sum to 1.0, `13` now scores `1.039772` inside the band — giving shipped `1`,
    capped `2`, delta `-1`, reduction `-1.0`. Four exact assertions now pin that
    outcome rather than only its sign, so a clamp or a sign flip fails on the
    number instead of degrading back to a vacuous `0.0 == 0.0`. The derivation
    of `0.155` is recorded in the test docstring: the priors are load-bearing and
    must not be tidied into round numbers.

## Revision 20 change log

Revision 19's Task 5 was implemented and its numeric block semantically
approved: `SEPARATION.min_auc = 0.70` and `min_lower_bound = 0.50` stand
unchanged. Two integration defects found at that review, plus one
interpretation frozen before its number is known.

83. **The role classifier could not consume its own producer's schema.** Task 5
    named predicate variables that do not exist in Task 7's frozen
    `census_positions.csv` — `exposure` for `exposure_primary_0.50`,
    `abs_root_value_stm` for `root_value_stm`, `count_abs_residual_over_1.25`
    for `would_clip_1.25`, `clipped_amount_at_1.25` for `clipped_amount_1.25` —
    and `max_abs_eligible_residual`, which the census **does not emit at all**.
    Two vocabularies for one quantity is how a selector silently reads a
    missing column at execution time, and the fifth name could never have been
    satisfied. A single CANONICAL FIELD CONTRACT now binds every predicate to a
    census column. Identity is expressed as `would_clip_0.5 == 0`, exactly
    equivalent to `max|residual| <= 0.50` because the clip rule is strict, and
    stated as a count because a residual maximum is not a column and must not be
    invented as one. A new test passes a row in the exact census schema through
    `classify_role` and asserts every field it reads is a member of
    `CENSUS_SCHEMA`.
84. **The prospective-target subset floor was not enforced where it binds.**
    `revisit_form` refused only an EMPTY subset, so a 3-row population could
    decide the form of the whole criterion on 2 dense rows — the exact vacuity
    the frozen 16-row floor exists to prevent, and the tests compounded it by
    exercising 4-row populations. The floor now binds inside the helper, and the
    boundary cases are tested at the real size: 12/16 dense -> `paired`, 11/16
    dense -> `candidate_only_floor`, 15 rows -> refusal naming
    `prospective_target_subset_below_floor`.
85. **The separation failure interpretation is frozen before the number is
    known.** A failure means "required A-vs-matched-control selectivity was not
    established", never "no effect exists". At the approved threshold the
    operating characteristics give 51.7% power when the true AUC equals 0.70, so
    a boundary miss is near a coin flip and carries no evidential weight against
    the mechanism. Freezing the wording now removes the opportunity to
    renegotiate it once the measured value is in hand.

## Revision 21 change log

Revision 20 fixed the role classifier but left two stale aliases in the
DOWNSTREAM consumers of the same contract. Naming-only; the approved numbers,
the AUC operating characteristics and the frozen failure interpretation are
untouched.

86. **Two gates still named a column the census does not emit.**
    `SEPARATION["statistic"]` and `EXPOSURE_CUTOFF_RULE["statistic"]` both read
    `exposure_at_cap_0.50`, which is absent from `CENSUS_SCHEMA`. Revision 20
    repaired the classifier and stopped there, so the defect simply moved
    downstream — any consumer indexing rows by the separation or cutoff
    contract would have missed the authenticated census entirely, and the
    cutoff is what defines the prospective target subset that R_min and the
    revisit-form criterion both stand on. The root cause was the contract being
    restated as a literal in six places. `PRIMARY_EXPOSURE_COLUMN` is now
    defined **once** and referenced by `CENSUS_SCHEMA`,
    `REQUIRED_CENSUS_FIELDS`, the target role predicate, `SEPARATION`,
    `EXPOSURE_CUTOFF_RULE` and `classify_role`; a test asserts all of them
    agree and that the retired spelling survives nowhere in `as_dict()`.
87. **`MATCHING` retained an undeclared `side` alias.** The census emits
    `side_to_move`; `variables` and `tolerances` said `side`. Unlike
    `abs_root_value_stm`, which is a legitimately DECLARED transform (the
    matcher pairs on magnitude, and `side_to_move` carries the direction), this
    one was an alias with no declaration — precisely the pattern the no-alias
    contract forbids. Both now use `side_to_move`, and a test asserts every
    matching variable is either a census column or a member of
    `derived_variables`, with `side` absent from both maps. Only
    `abs_root_value_stm` remains derived.

## Revision 22 change log

Revision 21's Task 4 was implemented and `seed20116` selected. Four
authentication defects were repaired at review; two survived into a second
review and are fixed here. No threshold, census policy or source choice moved.

88. **`black_checkpoint_sha1` / `red_checkpoint_sha1` made false claims.** The
    selected source is COLOUR-BALANCED -- each checkpoint plays black in 400
    games and red in 400 -- so there is no fixed colour assignment to record.
    The keys were retained for compatibility with a Task 4 binding contract that
    was itself factually wrong; compatibility with a false contract is not a
    reason to keep emitting a false field. They are REMOVED, and the record now
    carries `checkpoint_sha1s`, `anchor_checkpoint`, and `games_by_colour` with
    both colours counted for both roles. `colour_split` is renamed
    `games_by_colour` because it counted only black assignments, leaving the red
    side unverified -- exactly where an imbalance would hide. Both
    `a_as_black.games` and `a_as_red.games` are now cross-checked against the
    summary. Discovered by the per-replay check itself: the first
    implementation asserted a fixed pair and failed on game 1, which is what
    proved the alternation.
89. **Authentication did not bind the bytes that were consumed.** Files were
    hashed and then REOPENED to derive the exclusion sets and the census, so a
    file changing between the two reads would supply rows the recorded hash does
    not describe -- the artifact would carry a hash for content it never
    measured. Every pinned artifact is now read once into a buffer, hashed from
    that buffer and parsed from that same buffer; `forbidden_canonical_hashes`
    takes the authenticated payloads and REFUSES if none are supplied, and the
    census consumes the authenticated replay snapshot rather than reopening
    sidecars. Where the established length-delimited
    `fpu_provenance.replay_data_sha1` must read from disk, a CLOSING
    re-authentication runs after consumption and before any write, so drift is
    refused with no partial artifact. Also removed an eagerly-evaluated
    `dict.get` default that hashed the whole replay directory a second time and
    discarded the result.

## Revision 23 change log

Revision 22 closed the two prior authentication P1s. A third survived: the
snapshot contract covered the SELECTED universe but not the replay reservoirs
the forbidden evidence points into. Naming and authentication only; no
threshold, census policy or source choice moved. `seed20116` unchanged.

90. **Forbidden replay bytes were authenticated inconsistently.** The probe CSVs
    and manifests were byte-pinned, but `forbidden_canonical_hashes` then
    REOPENED each `replay_path` they reference without binding those bytes, and
    `forbidden_game_content_sha1s` reopened the A reservoir after its aggregate
    had been checked, with no closing verification. The canonical exclusion
    hashes are RECONSTRUCTED from those sidecars, so a replay change alters the
    exclusion set while every evidence-file hash in the record stays unchanged --
    the record would then describe an exclusion set that was never derived from
    the bytes it names. Three reservoirs are now pinned by count and by the
    established aggregate (`seed20115` 427d4ab6…, `seed35791` d36b01c0…,
    `seed40937` 80aa2068…), authenticated before any exclusion is derived and
    re-authenticated before anything is written, and a referenced replay outside
    every pinned reservoir is a hard refusal. Drift in a gate B, gate C or A
    replay is mutation-tested end to end and must leave no artifact.
91. **The JSONL/sidecar test never reached the validator it named.** Mutating
    the JSONL tripped the pinned-hash gate first, so the test would have stayed
    green with the cross-field comparison deleted -- it proved the hash check
    twice and the comparison never. It now repins the expected JSONL hash to the
    mutated bytes so authentication passes and the winner disagreement is what
    refuses; a separate test keeps the hash gate covered. Verified by deleting
    the comparison: the test now fails.

## Revision 24 change log

Revisions 22-23 hardened `freeze_source_universe`. The REPORT path predates the
snapshot contract and was never brought up to it. Narrow correction; no source
choice, threshold, geometry or scientific rule moved.

92. **The report bypassed every protection the freeze had gained.** It derived
    forbidden identities and census rows with no opening or closing
    authentication of the three forbidden reservoirs, no authentication of the
    selected universe, and called `enumerate_census` WITHOUT the snapshot, so it
    reread the sidecars it had never verified. Read-only is not the same as
    unauthenticated: figures handed to a human and used to choose a source are
    evidence, and evidence derived from unverified bytes is a guess. The report
    now runs the identical authenticated path -- payloads, reservoirs, selected
    snapshot into `enumerate_census`, `reverify_all_replay_sources` before
    returning -- and still writes nothing.
93. **The report left a post-selection inspection route open.** After
    `seed20116` was bound in tracked code, `--universe v16_production` would
    still have enumerated an alternative source. Preregistration exists to close
    exactly that door, so real reports now accept only
    `SELECTED_UNIVERSE["name"]`; fixtures remain available to the unit tests.
    Candidates 2 and 3 were never inspected.
