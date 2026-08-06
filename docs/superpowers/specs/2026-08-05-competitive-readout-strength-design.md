# Competitive Readout — Strength-First Experiment Design

**Created:** 2026-08-05 · **Revision 2:** 2026-08-06 · **Status:** DESIGN, not authorized to run
**Checkpoint:** unchanged `calib020_0001`
**Predecessor:** `docs/superpowers/2026-08-05-atlas-closeout.md` (tree-local search heuristics closed)

**Authorization state: NO GPU work is authorized by this document.** Every match, screen
and telemetry run named here requires its own separate written authorization naming its
exact scope, seed interval and game count. This spec defines *what* would be run and
*what would decide it*; it authorizes nothing.

**Nothing in this design has been executed.** Revision 2 therefore precedes all work it
governs, as required.

**Candidate 2's rule and preflight gates were reviewed and FROZEN on 2026-08-06**, before
any telemetry exists (§7.4). The colour-split statistic was demoted to descriptive at
freeze time and may not be promoted back.

### Changelog — what revision 1 got wrong

| # | Defect | Fix |
|---|---|---|
| 1 | Candidate 2 was frozen *after* Candidate 1 supplied its telemetry, so formula *selection* could still be data-influenced even with externally-sourced constants | §11 reorders: freeze Candidate 2 completely before Candidate 1 is authorized |
| 2 | Candidate 2 stacked argmax with Q — its non-override plies played deterministic argmax against a control sampling at `T=0.1` | §7.4 makes the control post-opening argmax, isolating Q against its actual fallback |
| 3 | A winning Candidate 2 was routed to policies 1 and 3, neither of which can receive it | §5, §8.1: research promotion bar; policy 3's target must be redefined before it can receive the rule |
| 4 | The formula did not exist, and preflight gates read "usually too sparse", "almost never", "unacceptably large" | §7.4 freezes the exact formula, constants, eligibility and numeric gates |
| 5 | "Q values" with no perspective contract, against `mcts.py:1122`'s `q = -child.q_value` | §7.2 freezes a two-field contract |
| 6 | "Score by agent identity" with no artifact schema, against checkpoint-keyed analyzers | §7.2 freezes the schema and the legacy-artifact rule |
| 7 | Claimed opening temperature was the only entropy source; in fact search and readout share `self.rng` | §7.1 gives evaluation search and readout separate streams and records the deviation |

---

## 1. Objective and primary endpoint

Improve the competitive playing strength of unchanged `calib020_0001` at exactly 400 new
simulations per move, by changing only the **final move readout** — the rule that picks
the played move from a completed search.

The primary endpoint is a balanced-colour, same-checkpoint head-to-head match. Diagnostic
position gates are not substitutes for strength and do not gate a candidate before a
strength run.

This line changes no search algorithm, search budget, tree mechanics, network, training,
or self-play behaviour. (It does change how evaluation *orchestrates* search randomness —
see §7.1's recorded RNG-stream deviation — which is why the sentence is scoped this way
rather than claiming "no search behaviour change".)

## 2. Standing against prior work

### 2.1 Not a do-not-repeat entry

Checked against all 47 entries of `docs/alphazero-value-search-experiment-ledger.md`:

- The readout is **post-tree**. It changes no selection value, no backup, no move
  eligibility and no prior. It is not tree-local in the closeout's sense, and is untouched
  by `c_puct` (#39), the three FPU formulations (#41, #43, #45), the v18 depth-2 backup
  (#46), or progressive widening (#47).
- It is not a calibration branch. No manifest, no objective, no training surface.
- No ledger entry addresses which move is played given a fixed tree.

### 2.2 Adverse precedent, recorded

`docs/superpowers/decisions/2026-05-19-reverted-closeout-experiments.md` records a prior
**readout override** in this repository — `closeout_selection_tiebreak`, `mcts.py:479-531`
— whose value gate was relaxed from `0.95` to `0.90`. The result: more overrides, but a
worsened closeout tail and worsened `td=2` quality, because "the added overrides included
false positives that disrupted otherwise-healthy closeout play."

This does not close Candidate 2. It is direct evidence that **overriding the visit leader
on a relaxed confidence gate can lose strength on this engine**, and it is the reason
this design chooses a deliberately conservative rule, keeps the visit leader as the
fallback, and retains Candidate 2's 64-game screen.

## 3. Established facts (measured from the code)

Each was read directly; none is quoted from a prior document.

### 3.1 The Python evaluation harness

| Fact | Location |
|---|---|
| `selection_mode` already supports `"argmax"` | `eval_runner.py:61` |
| `argmax` maps to temps `0.0/0.0`, hitting `select_move`'s deterministic branch | `eval_runner.py:79-95` |
| `--selection-mode` already exposed on the CLI | `eval_checkpoint_match.py:85` |
| Both agents are built from **one** `EvalConfig` | `eval_runner.py:107-108` |
| Evaluation search is **COLD** — fresh root every ply | `mcts.py:551` via `search()` → `search_with_root()` |
| Self-play advances the root instead (tree reuse) | `self_play.py:1168` |
| Readout: temperature `1.0` for plies < 20, then `0.1` | `mcts.py:164-166`, `1394-1420` |
| `add_noise=False` in evaluation | `eval_runner.py:115` |
| Same-checkpoint pairings return `None` for score rate, Elo and CIs | `eval_summary.py:67-75` |
| Summary records a single `selection_mode` for the pairing | `eval_summary.py:62` |
| Root-perspective Q is the **negation** of the child's stored value | `mcts.py:1122` |
| `MCTSNode` stores `visit_count` and `value_sum` only — **no second moment** | `mcts.py:244,259-261` |

**One RNG serves search and readout.** `self.rng` is consumed by prior-shuffle
(`mcts.py:1021`), PUCT tie-break (`:1155`), argmax tie-break (`:1407`) and temperature
sampling (`:1418`). `rng.choice` and `rng.random()` consume different amounts of
generator state, so **changing the readout changes the RNG state entering every
subsequent search**. A readout comparison run on the shipped harness would therefore not
be readout-only. §7.1 fixes this.

Consequences that shape this design:

- **Cold is the shipped evaluation regime.** A warm-tree match would rewrite the game
  loop and break comparability with the historical Elo numbers this experiment cites.
  This spec is cold.
- **`eval_summary` refuses comparative statistics on a self-match.** Passing two configs
  is not sufficient; agent identity must be decoupled from checkpoint path (§7.2).

### 3.2 Sampling behaviour

`select_move` and `server/mcts.js:selectMove` both sample proportional to `count^(1/T)`.

The table below is a **two-move illustration**: it gives the probability of playing the
runner-up *conditional on the choice lying between the leader (100 visits) and the
runner-up (80 visits)*, ignoring all other moves. With more visited moves the
unconditional runner-up probability is lower, but the qualitative ordering holds.

| Setting | Exponent | P(runner-up \| leader-or-runner-up) |
|---|---|---:|
| `T = 1.0` (Python, ply < 20) | `count¹` | ~44% |
| `T = 0.5` (server WS, medium) | `count²` | ~39% |
| `T = 0.25` (server WS, hard) | `count⁴` | ~29% |
| `T = 0.1` (Python ply ≥ 20; server REST) | `count¹⁰` | ~10% |

The existing readout is therefore not near-argmax in close positions even after ply 20.
All-ply argmax changes post-opening play as well as the opening, and must never be
described as an opening-only change.

### 3.3 The product server

| Fact | Location |
|---|---|
| Live client is WebSocket-primary, HTTP as fallback | `assets/js/ai/alphaZeroClient.js:2,197-200` |
| WS path uses `DIFFICULTY_PARAMS.moveTemp`: easy `1.0` / medium `0.5` / hard `0.25` | `server/index.js:41-46,523-541` |
| WS path does **not** support `deterministicMode` | `server/index.js:523-541` |
| REST path ignores `DIFFICULTY_PARAMS.moveTemp`; uses `easy ? 0.5 : 0.1` | `server/index.js:94-104` |
| REST path supports `deterministicMode`, default `false` | `server/index.js:64` |
| Budgets differ by difficulty: 100 / 400 / 800 | `server/index.js:43-45` |
| Cache key is `pegsHash:movesHash` only | `server/cache.js:72` |
| Cache is consulted before difficulty is applied | `server/index.js:79-83` |

Three defects, not one: the transports implement different policies for the same
difficulty; the deterministic override is unreachable from the transport real users hit;
and **the cache key omits difficulty, simulation budget, readout policy, custom
temperature and model identity**, so a 100-simulation `easy` result can be returned for a
later 800-simulation `hard` request at the same position. Caching a *sampled* final move
additionally makes repeated stochastic requests sticky.

This is **not** evidence for the research question and does not substitute for it: the
JavaScript search is a different implementation, `hard` runs 800 simulations rather than
400, and the live ONNX model (`MODEL_PATH` / `model.onnx`) has **not** been established
to be `calib020_0001`. Treat its identity as unknown until verified.

## 4. What argmax is and is not

Argmax maximizes the move's **observed visit-count estimate**. It maximizes playing
strength only insofar as higher visit counts imply higher true move quality. With 400
simulations over roughly 500 legal moves, and with the atlas's finding that deep
references are frequently unstable, that implication is not free.

Three consequences bind this design:

1. A null result is **not** automatically a harness defect.
2. A clear argmax loss is a significant finding and halts the line for investigation.
3. Argmax is never recorded as strength-proven by convention.

Equally, the question does not justify an 800-game match, because all-ply argmax mainly
removes a deliberate evaluation-diversity mechanism rather than introducing a search
improvement.

## 5. The four frozen readout policies

Distinct, frozen separately. Conflating them is the primary adoption risk in this line.

| # | Policy | Rule | Notes |
|---|---|---|---|
| 1 | **Product gameplay** | easy: intentionally exploratory · medium: explicitly chosen difficulty behaviour, likely retaining some temperature · **hard: deterministic argmax** | REST and WS share one policy function. Deterministic override reachable from both transports. **No Python result adopts directly into this policy.** |
| 2 | **Checkpoint strength tournament** | UNCHANGED: stochastic first 20 plies, near-argmax after | Preserves game diversity and historical comparability. **Must not become all-ply argmax.** |
| 3 | **Deterministic analysis / competitive mode** | all-ply visit argmax | For repeatability and strongest estimated move. Must not generate a checkpoint tournament without a separate opening-diversity mechanism. |
| 4 | **Self-play** | COMPLETELY UNCHANGED | Temperature, Dirichlet noise and warm-tree behaviour are training decisions, out of scope. |

**Policy 3 as written cannot receive Candidate 2.** It is defined as all-ply visit argmax;
a Q-informed post-opening rule is a different object. If Candidate 2 succeeds, policy 3's
adoption target must be **explicitly redefined in writing** before the rule can enter it.
This spec does not perform that redefinition.

Adopting policy 3 as policy 2 would destroy the instrument any future training work
depends on, and is prohibited.

## 6. Workstream 1 — product server (no GPU)

Independent of the research workstream and not blocked by it. **It does not substitute
for the research workstream and produces no strength claim.**

**Scope**

- One shared policy resolver consumed by both the REST handler and `computeBestMove`.
- Difficulty → (`nSims`, readout) defined in exactly one place.
- `hard` deterministic.
- `easy` and `medium` explicitly defined as difficulty behaviour, with the intended
  strength sacrifice stated rather than left implicit.
- Deterministic override accepted and honoured on both transports — and **the WebSocket
  client must send it**, not merely have the server accept it.
- **Cache correctness.** Preferred fix: cache *raw search results* keyed by state, model
  identity and simulation budget, then apply the requested readout afterwards. This also
  removes the sticky-sampled-move defect. Acceptable fallback: include the complete
  resolved policy identity — difficulty, budget, readout, custom temperature, model — in
  the cache key.

**Out of scope:** simulation budgets, the ONNX model, the JS search, the difficulty tiers.

**Verification.** Test the shared policy resolver directly. Transport parity **cannot** be
asserted as identical moves under stochastic settings without controlling randomness, so
parity is tested in deterministic mode or with mocked visit counts and RNG. Construct the
negative case — the check must fail when the two transports are given different policies —
rather than observing agreement from ambient state.

**Open item, blocking any product strength claim:** verify what `MODEL_PATH` /
`model.onnx` actually is.

## 7. Workstream 2 — research

### 7.1 Shared invariants

- Identical `calib020_0001` checkpoint bytes, verified by hash.
- Exactly **400 new simulations per move**, asserted, not merely recorded.
- **Cold search** — fresh root every ply.
- `add_noise = false`.
- Balanced colours; fresh, non-overlapping seed intervals per run.
- One evaluator instance per checkpoint per worker, reusing the existing path-keyed
  cache. Build one compiled evaluator and reuse it; rebuilding per agent or per game is
  the documented MLX Metal-exhaustion trap.
- Identical termination, move-cap and game-rule settings.
- **Separate RNG streams.** Evaluation search and move readout draw from separately
  derived streams, so a readout can never advance or perturb the generator state entering
  a later search. A test must assert this: invoking either readout leaves the search
  stream's state unchanged, with the negative case constructed. **Self-play is untouched.**
- **The only difference between agents is the final move-readout rule.**

**Recorded deviation:** separating the streams changes the shipped harness's RNG coupling.
Historical matches are therefore not reproducible game-for-game under this configuration.
This is a deliberate choice — without it the experiment is not readout-only — and the
comparability claim in this spec is narrowed accordingly: *every historical Elo number
cited by this experiment* was measured cold, and cold is preserved; exact RNG-level
reproduction of those runs is not claimed.

### 7.2 Required harness capability

Build only this. Do not build a general experiment framework.

**Agent identity.** Candidate and control are distinct identities even when the checkpoint
path is identical, so `eval_summary` computes comparative statistics instead of returning
`None`. The result artifact carries at minimum:

```
red_agent_id · black_agent_id · winner_agent_id
red_readout_config · black_readout_config
same_checkpoint: true
comparison_unit: "agent"
```

Checkpoint fields are retained for model provenance, but the winner's **experimental**
identity is never derived from `winner_checkpoint`.

**Legacy compatibility is explicit, not implicit.** Existing loss analyzers score by
checkpoint and would silently produce nonsense on same-checkpoint games. Use additive
optional dataclass fields or a separate result type, version the replay schema, and make
old analyzers **explicitly reject or route** agent-identity artifacts. Silent acceptance
is a defect.

**Per-ply telemetry contract.** For the top-two root children by completed visits:

```
move                        (row, col), canonical numeric order
completed_visit_count       int, > 0 required for the row to be usable
q_value_child_perspective   float, finite — as stored on the child
q_value_root_perspective    float, finite — equals -q_value_child_perspective
```

Both perspectives are recorded so the sign can never be inferred wrongly downstream
(`mcts.py:1122`). Values must be validated finite; a non-finite or absent value makes the
statistic `None`, never `0.0`. Statistics are read from the completed search **before**
readout and must not mutate any node.

**This telemetry supports only formulas over visit counts, Q means, and theoretical
bounded-return confidence.** An empirical variance-based LCB is not implementable without
a second-moment accumulator that `MCTSNode` does not have; adding one would change the
backup path and is out of scope for this design.

**Other requirements**

- Per-agent configuration bound to the correct identity across colour swaps.
- Replay provenance recording which readout occupied which colour in every game.
- **Fail loudly and fail closed** on configuration leakage, identity or colour
  mis-binding, or unequal simulation budgets. An unverifiable condition is a failure.
- **Constructed search-identity test.** At fixed root states with fixed streams, both
  readouts produce byte-identical `visit_counts` and `root_value`, differing only in the
  selected move. This is a per-position property — false across a game by construction —
  so it must not be tested at game level, and its negative case must be constructed.

**Baseline test count** comes from a measured `pytest` collect at the time of the work,
never from a number quoted in any document. Read exit codes from the process, not a pipe.
**Undefined statistics are `None`/null** — never `0`, never `false`.

### 7.3 Candidate 1 — 64-game all-ply argmax diagnostic

**A diagnostic, not a promotion match. It has no 800-game follow-up.**

**Control:** existing tournament readout — temperature `1.0` through ply 19, then `0.1`.
**Candidate:** visit-count argmax at every ply; deterministic canonical tie-break.
No Q, no search, no FPU, no tree changes.

**Purposes**

1. Validate candidate/control identity and colour binding end to end.
2. Validate equal budgets, RNG-stream separation, provenance and replay capture.
3. Sight the effect: grossly positive, near-null, or unexpectedly negative.
4. Measure wall-clock, so the 800-game commitment is costed before it is made.
5. Produce fresh root-child visit/Q telemetry for Candidate 2's **already-frozen**
   preflight.

**Interpretation**

| Outcome | Consequence |
|---|---|
| Large argmax win | Useful confirmation. **No 800-game follow-up.** Does not change policy 2. Supports policy 3 and informs policy 1. |
| Near-null | Plausible per §4. Inspect mechanics, then apply the frozen preflight with the anomaly recorded. |
| Clear argmax loss | **Halt.** Understand why visit leaders are unreliable before spending Candidate 2's budget. |
| Integrity failure | Fix the harness and rerun the same screen. Not a result. |

Replay analysis reports non-leader selections before and after ply 20 so a null can be
attributed. **Those numbers gate nothing.**

### 7.4 Candidate 2 — one frozen Q-informed post-opening readout

The only strength experiment in this document. **Everything in this section is frozen
before Candidate 1 is authorized.**

#### Match configuration — isolating Q against its actual fallback

| | plies 0–19 | plies ≥ 20 |
|---|---|---|
| **Control** | temperature `1.0` | visit argmax |
| **Candidate** | temperature `1.0` | visit argmax **+ frozen Q override** |

Both agents sample identically through ply 19, solely to supply match diversity. The only
difference is the post-opening override. Revision 1's control sampled at `T=0.1`
post-opening, which silently bundled the post-opening half of Candidate 1 into Candidate
2 — that is corrected here.

**This match configuration is an experimental control setup. It does not become the
checkpoint-tournament default (policy 2).**

#### The frozen rule — Hoeffding lower confidence bound

**FROZEN 2026-08-06**, before any telemetry exists. No constant, eligibility rule or
threshold below may change hereafter.

Let the two root children with the highest completed visit counts be the **leader** `L`
and **challenger** `C`, ties broken by canonical numeric `(row, col)` order. Using
root-perspective Q:

```
ε(n) = R · sqrt( ln(2/δ) / (2n) )        R = 2, δ = 0.05
     = 2 · sqrt( 1.84444 / n )

LCB_i = q_root_perspective_i − ε(n_i)

play C  iff  n_L ≥ 8  and  n_C ≥ 8  and  LCB_C > LCB_L
otherwise play L
```

**Provenance of every constant.**

- `ε` is Hoeffding's inequality for the mean of bounded observations.
- `R = 2` is the range of **backed-up MCTS values**, `[-1, 1]`. This is a property of the
  backup, not solely of the network: terminal values participate in backups too.
- `δ = 0.05` is the conventional confidence level.
- `n_min = 8` follows from a **preregistered judgement**, not from Hoeffding. We chose the
  usefulness requirement `ε(n) ≤ 1.0` — a radius no wider than half the value range — and
  `2·sqrt(1.84444/n) ≤ 1 ⇒ n ≥ 7.4 ⇒ n_min = 8` follows from *that choice*. The
  requirement is a judgement; only the arithmetic is forced.

**No constant is fitted to any TwixT data.**

**What this rule claims, and what it does not.**

1. `LCB_C > LCB_L` is a **conservative ranking heuristic**. It does **not** establish at
   95% confidence that the challenger is genuinely the better move — that would require a
   different comparison and sampling assumptions this setting does not satisfy.
2. `δ = 0.05` sets the **scale of the radius only**. It is not the match's statistical
   alpha, and it carries no repeated-decision guarantee across the thousands of positions
   at which the rule is evaluated.
3. **MCTS backups are adaptively sampled and correlated, not i.i.d.**, so Hoeffding here
   is a *principled unfitted radius*, not a valid guarantee. This limitation is the most
   important caveat on the whole rule.

It is chosen because it is conservative, derivable without TwixT evidence, implementable
from existing means and visits, strictly post-tree, easy to falsify, and independent of
any assumption that deeper search is truth — and because §2.2's precedent shows a relaxed
override gate losing strength on this engine.

Worked magnitudes: `ε(190) = 0.197`, `ε(100) = 0.272`, `ε(40) = 0.429`, `ε(8) = 0.960`.
With a 190-visit leader and a 40-visit challenger, the challenger must exceed the leader
by `0.232` in root-perspective Q to override. **The rule is expected to fire rarely; a
near-no-op is a legitimate outcome that closes Candidate 2 without a match.**

#### Erratum — display rounding, 2026-08-06

Two printed values in this section were misrounded. Both were caught by tests during
Task B1 implementation and both are corrected above.

| Printed | Correct | Exact value |
|---|---|---|
| `ε(n) = 2·sqrt(1.84445/n)` | `1.84444` | `ln(2/0.05)/2 = 1.8444397270569681` |
| `ε(40) = 0.430` | `0.429` | `0.4294694083467376` |

**NO EXECUTABLE CONSTANT CHANGED.** `eval_readout.py` has always computed
`_HOEFFDING_NUM = math.log(2.0 / DELTA) / 2.0` directly from the frozen `R` and `δ`; it
never contained a literal `1.84445`, `1.84444` or any worked magnitude. The two values
above are display text for human reading, and both appear only in prose.

**The exact formula `ε(n) = R·sqrt(ln(2/δ)/(2n))` with `R = 2` and `δ = 0.05` is
authoritative.** Where a rounded figure ever disagrees with it, the formula wins. The
frozen rule — formula, `R`, `δ`, `n_min = 8`, top-two eligibility, canonical `(row, col)`
tie order — is exactly as frozen on 2026-08-06 and is untouched by this erratum. The
`0.232` override gap and the `n_min` boundary (`ε(7) = 1.0266 > 1.0 ≥ ε(8) = 0.9603`)
are both unaffected.

`tests/test_eval_readout.py` now pins this: one test compares `hoeffding_radius(n)`
against `R·sqrt(ln(2/δ)/(2n))` computed from the frozen constants at `rel=1e-12`, and a
second asserts the printed display constant remains a faithful rounding of the formula —
so spec text and code cannot drift apart again without a test failing.

Rejected alternative: KataGo-style empirical-variance LCB. It requires second-moment
telemetry `MCTSNode` does not store, which would mean editing the backup path.

#### Preflight — frozen numeric gates

Applied to Candidate 1's captured telemetry. Descriptive evidence only; **the rule above
is already frozen and cannot be revised in response to what the preflight shows.** Atlas
rows are not used, and 3,200/6,400 results are not treated as truth.

**Preflight population, defined exactly.** The primary statistics are computed over:

- **Only post-opening turns (`ply ≥ 20`) belonging to Candidate 1's argmax agent** — that
  agent is Candidate 2's actual post-opening control, so this avoids contaminating the
  gates with Candidate 1's `T=0.1` control turns.
- **All such turns in the denominator.** A turn where the rule is ineligible (fewer than
  two children with `n ≥ 8`) counts as **"no override"**, not as an excluded row. Rows
  never disappear from the denominator.
- **Formula evaluation uses the raw completed-search telemetry captured before readout**
  (§7.2), with no mutation.

Turns outside this population may be reported descriptively but **must not affect any
frozen gate**.

| Statistic | Automatic stop | Kind |
|---|---|---|
| Override rate over the population above | `< 0.5%` → close | stop |
| Override rate over the population above | `> 15%` → close | stop |
| Share of all overrides falling in a single game | `> 50%` → close | stop |
| Rows where `q_value_root_perspective` is `None`/non-finite | any → halt, fix telemetry | stop |
| Colour split of overrides | — | **descriptive** |
| Override rate by ply bucket | — | descriptive |
| Distribution of `n_C` at override | — | descriptive |
| Per-game override count distribution | — | descriptive |

The three rate/concentration thresholds are **pre-registered judgement bounds chosen
before any telemetry is observed.** They are not derived from data, and that is stated
rather than disguised.

- **`< 0.5%`** — below this, the rule does not have enough observed reach to justify a
  64-game screen plus an 800-game match under this project's deliberately conservative
  spending policy. **This is a preregistered spending judgement, not a bound proving that
  a rare override cannot be decisive.** It is consistent with §7.4's own position that a
  rare move may matter disproportionately.
- **`> 15%`** — above this the rule is no longer the conservative occasional override this
  design hypothesized. Closing is a **mechanism/scope decision, not a prediction that it
  would lose.**
- **`> 50%` in one game** — deliberately permissive. It fires only where the aggregate
  override rate is mostly describing a single abnormal trajectory.

**Colour split is descriptive only, by decision.** A colour-symmetric formula can
legitimately fire at different rates by colour, because position, branching, visit and Q
distributions differ by colour; trigger imbalance is not itself evidence of strength harm.
At low override counts a `25/75` split is also noisy enough to reject a balanced mechanism
by chance. The concern that matters — convincing playing-strength harm in one colour — is
already tested by §8.1's colour-specific 95% upper bounds over 400 games per colour.

#### Strength evaluation

64-game screen (§8.2) on a fresh seed interval, then one 800-game decisive match (§8.1) on
a further fresh interval. Screen seeds are **not reused in, and not counted toward, the
800.** Same frozen configurations throughout; no top-up, no rerun on the basis of a result.

## 8. Statistics and decision rules

Bounds assume approximately Bernoulli variance (0.25 per game). Draws reduce variance
slightly, so the bounds are mildly conservative. State caps historically run about
1–2.5% in 800-game matches.

### 8.1 The 800-game decisive match

400 games per colour, fresh seed interval, no top-up.

```
n = 800   SE ≈ 0.0177   95% half-width ≈ 0.0347
promotion requires observed score rate ≳ 0.535  ≈  +24 Elo
~80% power at a true effect of ≈ 0.550          ≈  +35 Elo
```

**This is the research promotion bar, written down before the run.** It is not a ship
bar: no deployment target exists for a Q-informed rule (§5), so a significant result
promotes the rule for further work, it does not ship it.

Promotion requires all of:

- **Primary:** draw-inclusive score rate with a 95% lower bound above 50%.
- **Colour safety:** reject only if either colour's own 95% **upper** bound falls below
  50% — at 400 games per colour, an observed colour score at or below about `0.451`
  (≈ −34 Elo), i.e. convincing one-sided harm. No colour-gap veto.
- **Zero-tolerance integrity** (§8.3).
- **Search-identity evidence** (§7.2) confirming only final move selection differed.

**Secondary, reported but not decisive:** decisive-only score rate and state-cap count.
Secondary because excluding draws biases the comparison if the candidate changes draw
propensity.

**What a significant result means.** The frozen post-opening Q rule is validated relative
to the Python visit-argmax baseline, cold, at 400 simulations. It becomes eligible to
update the post-opening portion of policy 3 *after that policy's adoption target is
explicitly redefined in writing*, and eligible for a separately scoped JavaScript/product
transfer test. **It is not adopted into policy 1, and it does not change policy 2.**

### 8.2 The 64-game screen

Used by both candidates, on fresh seeds each time. It may **only stop**; it can never
promote, and no success decision is taken from it. Because it has no early-success path
it cannot inflate the probability of a false success, so no alpha-spending boundary is
required. The rule is frozen before the run regardless.

Unit tests establish mechanics; the screen catches gross real-model harm and operational
failure. No separate smaller smoke is needed unless the implementation cannot be safely
exercised in unit tests.

Stop for any of:

- Any zero-tolerance condition (§8.3).
- A 95% score-rate interval entirely **below** 50% — at n=64, an observed score at or
  below about `0.378` (≈ −87 Elo).

**Not required at 64 games:** 55% overall, or 45% per colour. That gate would have
discarded roughly two of every three candidates capable of clearing the 800-game bar,
while still advancing about one null in five.

**"Stop" differs by candidate.** For Candidate 1 — a diagnostic with no 800-game path — a
futility trigger means **halt and investigate** per §7.3; a clear argmax loss is a
finding, not a rejection. For Candidate 2 it closes the readout line.

### 8.3 Safety, frozen numerically

**Zero tolerance — any occurrence stops the run immediately:** illegal move, crash,
`unknown_error`, agent-identity leak, configuration leak, simulation-budget mismatch,
non-finite telemetry where a value is required.

**Reported, with any automatic stop threshold frozen in the run's own authorization:**
state-cap count and rate, wall-clock runtime, termination-reason distribution.

**There is no discretionary safety review after a significant score.** A condition either
stops the run by a pre-frozen rule or it is descriptive.

## 9. Role of diagnostic gates

The historical A/B/C/D probes measure root value at frozen positions. Neither candidate
changes the search tree or the root value, so **the probes are mathematically invariant
under both candidates**. They are not overridden by a strength result; they cannot fire.

Diagnostics retain force only through §8.3's zero-tolerance list.

## 10. Provenance

For every run, retain exactly what reproduces the result:

- Git commit and worktree state.
- Checkpoint path and cryptographic hash.
- Complete control and candidate configurations, including both RNG stream derivations.
- Seed interval and game count, and the intervals of every prior run, to prove disjointness.
- Per-game colour and configuration assignment **by agent identity**.
- Results, winner agent id, termination reasons, replay paths.
- Simulation-budget telemetry.
- Summary statistics with confidence intervals.

**No reservoir, selector, corpus sizing, classifier or multi-stage qualification protocol
is required or permitted.** That apparatus exists and is qualified; nothing here consumes it.

## 11. Routing

```text
Workstream 1 — product, no GPU (independent, unblocked)
  one shared policy resolver → hard = argmax
                             → easy/medium explicitly defined
                             → deterministic override on both transports, client sends it
                             → cache keyed by state+model+budget, readout applied after
                             → verify MODEL_PATH identity

Workstream 2 — research (each GPU step separately authorized)
  implement harness: agent identity + per-agent config + split RNG + extended capture
        │
        ▼
  VERIFY the implementation matches the ALREADY-FROZEN Candidate 2
  formula, constants, eligibility, preflight thresholds and stop
  rules (frozen 2026-08-06, §7.4).  Verification only — nothing here
  may alter the freeze.
        │
        ▼
  64-game all-ply argmax DIAGNOSTIC  (never promoted, never 800)
        ├── clear loss ──────────► HALT and investigate
        ├── integrity failure ───► fix harness, rerun screen
        └── win or near-null ────► continue
        │
        ▼
  apply the ALREADY-FROZEN preflight to Candidate 1 telemetry
        ├── any stop threshold hit ──► close readout line → revisit training
        └── pass
        │
        ▼
  64-game screen, fresh seeds, not counted toward the 800
        ├── stop ────────────────► close readout line → revisit training
        └── continue
        │
        ▼
  one 800-game decisive match, fresh seeds
        ├── null / negative ─────► close readout line → revisit training
        └── significant win ─────► research promotion (§8.1); policy 3's
                                   adoption target must be redefined in
                                   writing before the rule enters it
```

A candidate that fails is closed. It is not rescued with adjacent thresholds, a parameter
sweep, or a second formula. **If both close, the readout and search line is finished and
the next work is a separately scoped training project.**

## 12. Prohibited

- Warm trees in evaluation.
- Any proxy corpus, reservoir, selector or sizing protocol.
- Any coefficient grid or sweep, or a second Q formula after Candidate 2 closes.
- Stacking Candidate 1 and Candidate 2 in one agent.
- Revising the formula, its constants, its eligibility rule, or the preflight thresholds
  frozen on 2026-08-06 — including promoting the colour-split statistic back to a gate.
- Changing the checkpoint-tournament default to all-ply argmax (policy 2).
- Adopting any Python result directly into product policy 1.
- Any change to self-play, the network, training, or tree-local search.
- Reusing atlas rows, or treating 3,200/6,400 results as truth.
- Reusing screen seeds in the decisive match.
- Any GPU run without its own separate written authorization.

## 13. Open risks, recorded before the work

1. **The frozen Hoeffding rule may fire too rarely to matter.** Its radii are
   conservative by construction. The `< 0.5%` preflight floor exists to close it cleanly
   in that case; closing without a match is a correct outcome, not a disappointment.
2. **Hoeffding's i.i.d. assumption does not hold for MCTS backups.** The bound is a
   principled unfitted radius, not a guarantee.
3. **Candidate 2's plausible effect may sit at or below the 800-game detection floor.**
   Externally reported readout gains of this kind are in the tens of Elo.
4. **§2.2's precedent is adverse.** A prior override on this engine lost strength when its
   gate was relaxed.
5. **The live ONNX model's identity is unverified.** No product strength claim until it is.
6. **Python and JavaScript searches differ**, budgets differ (`hard` = 800), so a Python
   result informs but does not establish a product decision.
7. **A near-null Candidate 1 leaves opening/post-opening attribution ambiguous**, because
   all-ply argmax changes both. The replay non-leader counts mitigate this at no cost, but
   they are descriptive.
8. **RNG-stream separation is a real deviation** from the shipped harness; historical runs
   are not reproducible game-for-game under it.
