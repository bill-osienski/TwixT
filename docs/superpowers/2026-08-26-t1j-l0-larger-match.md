# L0 — the 64-game larger match, PREREGISTERED and QUALIFIED

**Date:** 2026-08-26 · **Status:** FROZEN, NOT EXECUTED. **No model, no JVM, no RNG, no move, no
game, no seed drawn, no execution gate, no push.** · Local, unpushed.
**The 64-game execution remains separately unauthorized.**

Basis: `main` @ `72ade5a`, clean. Evidence: `evidence/2026-08-26-t1j-l0-larger-match/`.

---

## What L0 is

The canonical screen returned joint `IN_BAND` with the strong endpoint at 7.0/8, which *permits* a
larger match. The screen answered a **band** question and was allowed to stop early once the answer
was forced. L0 asks a **measurement** question — what is T1j's score rate at `mdPly` 6 against
`calib020_0001` — and an early stop biases a rate even where it cannot bias a band decision.

So the screen's decision machinery is deliberately **not reused**. `l0_match_rules.may_stop_early`
is a constant `False`, `EARLY_STOP is None`, and the frozen plan names the five screen functions
that must never be called on an L0 run — with a test asserting each still exists, so the prohibition
cannot rot into a reference to nothing.

## The design

**8 frozen openings × 2 colour arms × 4 repetitions = 64 games**, all at `mdPly` 6, 32 per colour
arm, 4 per cell across 16 cells. Seed = `202613000 + task index` in frozen order.

Every parameter shared with the screen is **read from the screen's frozen plan**, whose sha256 is
the one `e4_screen_runner` already pins — openings, opening plies, reference identity, `our_settings`
(400 sims), ply cap 280, scoring, per-ply binding, abort rules, durable format. "Unchanged" is
checkable only if there is one source; retyping eight openings would create a second that looks
identical until it isn't. A control that retypes them inline is rejected.

Frozen: plan `d5119de264b6a3de…`, task digest `193d66bf5f1e4dca…`, 64 tasks.

## The seed block

`[202613000, 202613064)` — **reserved, unspent, nothing drawn.** Reserved only *after* proving it
disjoint from every category (`02`, `03`):

| category | size | overlap |
|---|---:|---:|
| ACCOUNTED / EXPOSED / RETIRED / TEST_ONLY / CONSUMED | 3040 / 65 / 32 / 100 / 1 | **0** each |
| the frozen plan's `prior_intervals` and its own `seed_block` | 2560 / 32 | **0** |
| **every derived RNG stream** of all of the above (256 vs 12596) | — | **0** |

The stream check matters on its own: a shared search or readout stream would correlate an L0 game
with an already-played one even with distinct seeds. It is XOR arithmetic — no generator is built.

The block is registered in `ACCOUNTED_SEED_INTERVALS` as **reserved**, not exposed: nothing has been
drawn from it, so it is still executable, and a test asserts that. `rng_witness` refuses it, because
a witness here would spend the match before it ran.

## The interval — a valid bound first, a nominal one alongside

Both are closed form, so the numbers are reproducible from the recorded scores with no resampling
seed to record, lose or change. A control that inserts a generator into either is rejected.

**Primary: Hoeffding.** Half-width `sqrt(ln(2/α)/(2n))` = **0.16977** at α=0.05, n=64. It assumes
independence and boundedness in [0,1] and *nothing else* — not a Bernoulli outcome, not identical
distributions. Both matter: a game scores 1.0/0.5/0.0, and the eight openings are eight different
distributions. Coverage is at least nominal by construction. It is wide, and that width is the
honest price of a distribution-free guarantee over 64 games.

**Secondary: Wilson, reported as nominal and approximate.**

> **A previous version of this card and module called Wilson "provably conservative". That was
> wrong**, and wrong in a way worth recording, because the argument looked like a proof. It showed
> that draws lower the variance — `Var(X) = p(1−p) − 0.25·P(draw) ≤ p(1−p)`, which is true — and
> then asserted **coverage** from it, which does not follow. A variance inequality is not a coverage
> statement. Wilson's own coverage oscillates below nominal regardless of draws: at n=64 its exact
> coverage is **87.97% at p=0.002**, and across p=0.001…0.999 it is below 95% at **43.3%** of
> values, including **94.01% at p=0.5** — squarely in the plausible region. My "exhaustive" test
> enumerated *observed compositions*, which says nothing about repeated sampling, and treated the
> observed draw count as the true draw probability.
>
> The coverage figures above are computed exactly in the test suite, not quoted.

`variance_deficit` is kept and **demoted**: a descriptive property of the observed mix, never a
coverage claim.

## What the report may and may not say

Primary: **overall T1j score rate over 64 games with a 95% Hoeffding bound** (Wilson is reported
alongside, nominal only — see above). Everything else —
per-opening (8 games), per-colour (32 games), runtime — is **descriptive, carries no interval, and
supports no comparison between cells.** `FORBIDDEN_CLAIMS` lives in committed code, not only in this
card, so a reporting script cannot claim what the protocol forbids without editing a module; a
control that deletes the Elo prohibition is rejected.

Stated in the plan, because they bound the result:

- **The estimand, written down before any data exists:** the equally weighted mean, over the 16
  opening/colour cells, of T1j's expected score at `mdPly` 6, the expectation being over **engine
  randomness only**. The design is balanced (4 × 16), so the plain mean of the 64 scores *is* that
  cell mean. It is **not** the expected score against TwixT openings in general — the eight openings
  are part of the estimand, not a sample from a population of openings.
- **Independence is a model, not a measurement, and the interval is conditional on it.** What was
  verified is that the 64 tasks derive 128 distinct generator streams colliding with nothing used
  before — that rules out accidental stream *reuse*. It does not establish statistical independence:
  the seeds are fixed consecutive integers, the derivation is a fixed XOR, and T1j seeds its own
  `Zobrist` table from an unseeded `Random` per process, which this design neither controls nor
  observes. So the figure is a **95% bound under an independence model**, and both the report and
  the plan say so in those words. Claiming the games "are independent" is in `FORBIDDEN_CLAIMS`.
- The games are also **not identically distributed** — openings differ — which is precisely why a
  binomial method cannot be primary.
- The 4 repetitions differ only in our agent's seed. That would isolate reference-agent sampling **if
  T1j were deterministic** — and E3a established determinism for one position at one ply, explicitly
  declining to prove it generally (`Zobrist` self-seeds from an unseeded `Random` per process). **T1j's
  contribution to within-cell variation is unknown, not zero.**
- **No Elo, no absolute placement.** T1j is uncalibrated; 64 games narrow the interval on this design,
  they do not widen what the design covers.

## The plan no longer contradicts itself on aborts

The frozen plan previously **imported the screen's `abort_rules`**, which include a *statistical*
abort — more than half of an endpoint cap-terminating makes it INCOMPLETE — while its own protocol
said only integrity failures may terminate early and banned the screen's incompleteness machinery.
A runner would have had two conflicting authorities.

L0 now carries **`L0_ABORT_RULES`: integrity only** (engine divergence, incomplete T1j depth, illegal
or null move, postcondition failure, artifact identity mismatch, seed outside the block or reused,
failed durable write). `NOT_ABORT_RULES` names what is deliberately excluded so it cannot be
reintroduced quietly.

Cap-heavy results are handled at **reporting** time, preregistered, with all 64 games always played:

| caps | outcome |
|---|---|
| > 32 (more than half) | `reported: False`, `CAP_SATURATED_NO_RATE` — a rate over mostly-unresolved positions measures the cap, not the players |
| 1–32 | rate reported, `cap_warning: True`, count in the summary |
| 0 | normal |

## The reporter checks the schedule it is given

`match_report` verified each result against the task list it was handed, but never that the **task
list itself** was the frozen design: renaming an opening inside it and supplying matching results
produced a report about an invented cell. The digest lived in `l0_match_plan`, *above* the reporter,
so the reporter could not check it — and a reporter that cannot check the schedule will report on any
schedule.

The frozen identity (`L0_TASK_DIMENSIONS`, `l0_task_digest`, `L0_TASK_DIGEST`) now lives in the
**rules layer**, which the plan imports, so `bind_results` verifies the canonical digest before
binding anything. `l0_match_plan` re-exports it, and an AST check asserts the rules layer does not
import the plan back.

## Scores are type-safe and the reporter never raises

Validation compared `float(got)` to the expected score, so **`True` and `"1.0"` were accepted as
wins** (`float(True) == 1.0`) and **`None` or a list raised an uncaught `TypeError`** — a traceback
instead of a refused report. `t1j_points` must now be a finite JSON number, with `bool` rejected
first because `isinstance(True, int)` is `True` in Python. Behind that sits a fail-closed backstop
converting any unforeseen exception into a refusal.

Nine malformed values are refused (`True`, `False`, `"1.0"`, `None`, list, dict, `nan`, `±inf`) and
`int`/`float` still accepted. A hostile-record test replaces each of five fields with each of eight
absurd values and asserts the reporter **never raises**.

**The fail-closed boundary now wraps the whole binding path, not just one call.** It guarded
`validate_result` only, so malformed shapes escaped as tracebacks *before* reaching it: a `None`
result raised `AttributeError` at `r.get("task_id")`, a non-dict task raised `TypeError` building
`by_id`, and task_ids of `None` and `"alien"` raised `TypeError` sorting mixed types. Container and
`task_id` types are now validated up front — so every later set operation and sort is on strings —
with an outer `try` around the entire bind as the last resort. Ten shape cases are refused with
actionable messages, and a fuzz crosses eight hostile values across six container positions
asserting the reporter never raises.

> **The backstop was unreachable, and I only noticed because a control failed to bind.** With the
> specific type checks in place, nothing in the hostile-record test ever got that far — so removing
> the backstop changed no test result. A gate no test reaches has not been shown to work. It is now
> exercised directly by monkeypatching `validate_result` to raise an unanticipated `KeyError`, plus a
> companion test proving the backstop does not swallow ordinary validation messages.

## Cap and win now match the production loop

`e4_screen_runner.play_task` checks `state.winner()` **before** the ply cap, so a natural win on ply
280 is a **win**, and a cap records `winner=None`. My validator equated `plies == 280` with a cap:
it rejected the legitimate win at the cap and accepted an impossible cap carrying a winning colour.
Now: **cap ⟹ ply 280 and no winner; win ⟹ plies 1–280 and a colour.**

The control for this is **behavioural, not a source grep**. An earlier version compared character
offsets of two strings inside `play_task` — matching prose would have satisfied it, the same defect
class the AST check avoids. Instead the tests now drive **the real `play_task`** with a synthetic
state at ply 280: winner `"red"` → `terminal_reason="win"`, `t1j_points=1.0`; winner `None` →
`"cap"`, `0.5`; winner `"black"` → `"win"`, `0.0`. All three assert `agents_built == 0`, and each
outcome is round-tripped through `validate_result`, so whatever the runner can emit at the cap, the
L0 validator accepts. An injected defect that reorders the runner's own cap/winner check is rejected.

## Results must be results, not names

`match_report` now takes the **canonical tasks**, not a list of IDs, and binds every result to its
task. It validates the score against the one implied by the **winning colour** and the terminal
reason, `plies ∈ [1, 280]` and consistent with the cap, `seed`/`endpoint`/`mdPly`/`anchor_colour`
matching the schedule, and 16 cells of 4. **`opening` and `colour_arm` are read from the task, never
from the row**, so a result cannot misreport its cell.

The payload that exposed this — all 64 canonical IDs with `t1j_points=0.25`, `plies=-7` and invented
cells — was accepted before and produced a rate and an interval. It is now refused.

Two further defects surfaced while fixing it:

- **`score_game` compared `winner` to the string `"t1j"`.** The screen's own records store `winner`
  as a **colour**, with `t1j_points = 1.0` iff the winner is the anchor's colour. No recorded row has
  ever contained `"t1j"`. A test now replays all 24 recorded screen games through `score_game` and
  reproduces their `t1j_points` exactly.
- **Recorded rows carry no `opening`/`colour_arm` at all**, so the old grouping would have raised
  `KeyError` on real data. Taking the cell from the task fixes both problems at once.

## Two defects in my own control harness

Both were caught by controls that *failed to bind*, not by the code under test:

**1. A same-length edit was silently ignored.** `N_REPS = 4` → `N_REPS = 3` left the file the same
size, and CPython invalidates a `.pyc` on `(source mtime in whole seconds, source size)` — so the
stale bytecode stayed valid and the test ran the **unmodified** module. The control reported
`NOT REJECTED` for a defect that was never applied. The harness now purges `__pycache__` and runs
with `PYTHONDONTWRITEBYTECODE=1`.

**This forced a re-check of the controls already published with `72ade5a`.** Re-run under the
hardened harness: **11/11 still rejected, baseline 182** (`07`). The published claim stands — but it
was re-verified, not assumed.

**2. The harness did not restore on a crash.** An exception between "inject" and "restore" left an
injected defect sitting in `e4_screen_command.py` — a published module — until `git status` caught
it. Restoration is now in a `finally`, and it proved itself later in this same round: a stale control
anchor raised mid-run and the tree came back clean.

Two L0 tests also failed to bind and are fixed rather than reworded: a tampered-plan control was
masked by the file-hash check (the digest is now isolated by re-pinning the hash, which is the
realistic failure — regenerate, update the hash, forget the digest), and a "parameters are read, not
retyped" test compared two files without ever exercising the builder (it now feeds a modified source
and watches the output move).

**A third recurrence of an old trap.** Three of my checks grepped raw source for a phrase and matched
**my own prose** — including a test that searched for "conservative" and hit the passage that *bans*
the word. All three now check meaning: the ban in `FORBIDDEN_CLAIMS`, the docstring's explicit
disclaimer, and the plan's `corrected_claim` field.

## Controls

**Thirty-one injected defects, each restored and hash-verified, all thirty-one rejected**, baseline 110
(`05`, `06`): early stop reintroduced · seed block overlapping the spent screen block · repetitions
dropped to 3 · an interval starts resampling · **Wilson promoted back to primary** · **Hoeffding
half-width loses its log term** · endpoint drifts to the weak depth · Elo prohibition deleted · `rep`
leaves the digest dimensions · loader stops checking the digest · a partial match reported anyway ·
parameters retyped instead of read · **results bound by name only** · **a result misreports its cell**
· **`score_game` back to comparing `winner` to `"t1j"`** · **cap policy silently reports a rate** ·
**the screen's abort rules imported again** · reporter stops verifying the canonical digest · a cap
carries a winning colour · a win on the cap ply rejected again · independence asserted as fact ·
estimand deleted · `t1j_points` back to bare `float()` · a string score coerced to a win · non-finite
scores allowed · the fail-closed backstop removed · **the runner itself reordered to decide the cap
before the winner** · shape checks skipped · the outer boundary removed · the results-container type
unchecked · the `task_id` type unchecked.

> **A pattern in the controls worth naming.** Three times a control failed to bind because a *blanket*
> already caught what the specific check catches — first the type-safety backstop, then the two
> container-type checks. Defence in depth is right, but it means an individual guard cannot be proven
> by a "never raises" test: it has to be proven by the **refusal message** it produces, which is what
> those controls target now. A layered gate needs a layered control.

## What L0 does NOT build

No execution harness, no runner variant, no command, and no gate to enable. L0 is the
preregistration and its qualification. Wiring the frozen plan to the qualified runner, binder and
recorder is the next step and is **not** authorized here.
