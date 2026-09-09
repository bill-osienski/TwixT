# D1″ RAN ONCE — outcome **NO_GO**. The D1 fragmentation line is CLOSED.

One invocation of `analyse_search_suppression` through its checked public entry, on the sealed
221-position record, exit **0**, 12.3 s. The acquisition manifest was verified **immediately before
use** (19 of 19) and again after (19 of 19): the record is unchanged. No model, no JVM, no query, no
seed action, no registry or gate change. All six gates False.

## The verdict

| | |
|---|---|
| **outcome** | **NO_GO** |
| T (matched search suppression) | **−0.0232** |
| threshold | 0.15 |
| empirical stability interval | **[−0.0782, 0.0124]** |
| undefined replicates | 0 |
| support floors | **met**: 21 cells, 62 positions, 54 controls, 17 games |
| positions validated | 221, 10 observables each |
| PRNG | PCG64, seed 20260908, B = 10,000 |

**The point estimate is slightly NEGATIVE**, so what excess there is runs toward the controls, not the
fragmentation positions. The interval spans zero and the effect is an order of magnitude below the
preregistered threshold. **NO_GO on every ground at once.**

## What the descriptive reports say, and what they do not

| | positions | controls |
|---|---|---|
| rows | 101 | 54 |
| `ss` rate | **0.0099** (1 row) | **0.0370** (2 rows) |
| complement, `rank_raw ≤ 5` | 29 (28.7%) | 23 (42.6%) |
| `readout_overrode_leader` | **0 of 101** | **0 of 54** |
| strict variant `ss0` | T = 0.0 | |
| `created_threat` cohort | T = 0.0 over 12 cells | |

**Search suppression as defined is RARE in both roles** — three rows out of 155 — and 18 of the 21
common-support cells show exactly zero difference. The measured quantity is small in an absolute
sense, not merely small relative to its controls.

🔑 **The readout never once overrode the visit leader** in either role. The second mechanism the plan
named — the readout declining the move the search preferred — **did not occur at all** in this
cohort. That is a descriptive fact about these 155 rows and is not licensed as a general claim about
the readout.

⚠ **The complement is smaller at positions (28.7%) than at controls (42.6%).** This is reported
because a reader needs it to interpret the result, and it is exactly what the amended plan predicted
would happen mechanically: fewer policy-eligible rows leave less room for `ss` to occur. It decides
nothing, and no statistic here was conditioned on it.

## What this establishes, stated carefully

In the frozen §13 development cohort, positions our model marks as mover-fragmentation weaknesses do
**not** show a larger matched excess of search suppression than their controls; the point estimate is
slightly negative and unstable under whole-game reweighting. Combined with the low absolute rate, the
frozen conclusion is that **this mechanism is not supported strongly enough to spend the sealed half
on**.

**It is not a null**, and it does not show that our search never discards a move our policy ranked
highly: the interval is an **empirical stability interval**, not a null-hypothesis confidence
interval, and one position did show suppression. What it forecloses is *this* hypothesis at *this*
threshold on *this* cohort.

## Consequence, per the frozen rules

**NO_GO closes the D1 fragmentation line.** No confirmation acquisition is proposed, the confirmation
half stays sealed and unopened, and no training intervention follows. Both mechanisms the line
proposed have now been measured on the development half and neither cleared its bar:

- **D1′** (policy-rank disagreement): T = 0.0695, below 0.15, interval spanning zero.
- **D1″** (search suppression): T = −0.0232, below 0.15, interval spanning zero, base rate ~2%.

Anything further is a **new preregistration**, not an extension of this one.

## One fault of mine in the packaging

My first process check grepped for `java|t1j` and matched **this own command**, whose text contains
both, so the record briefly claimed a JVM-like process was running. A pattern that matches the
observer is not an observation; the recorded check matches the `bin/java` binary. The analysis itself
was untouched by this.
