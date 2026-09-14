# H3 PILOT — THE INCUMBENT'S IDENTITY: PROVENANCE REPAIR

**RESULT OF RECORD, 2026-09-14.** Every figure was read from the runs packaged
here at the moment of recording.

**NOTHING RAN.** No game, no seed drawn, no model loaded, no JVM. All EIGHT
gates False. Block `[202624000, 202624040)`: accounted 40 / exposed 0 /
retired 0 / test-only 0, none in `CONSUMED_SEEDS` — **unspent**. Push held.
Pilot execution remains a separate decision and is not requested here.

## What was authorized

> "Add the identity barrier before the pilot runs… Derive H3's identity from the
> frozen argmax configuration, verify it type-strictly before opening results,
> and write the full identity and `selection_mode` into the durable header. Test
> that the recorded identity matches the configuration passed to the real
> builder, with refusal controls for drift."

and then, on review of the first attempt:

> "**the identity is not read from the same config object passed to the
> builder.** … construct the argmax config once before opening outputs, derive
> and verify the identity from **that object**, and pass that same object into
> the production seam. A control should make the config handed to the builder
> differ from the recorded one and require refusal before play."

Commits `9b5beda..3a52f0e` (5). UNPUSHED.

## 1. THE GAP

H3 passed `identity={}` and wrote neither the identity nor `selection_mode` into
the durable header. Nothing would have played wrong — the configuration is
pinned per task and checked by the pre-run builder call — but the results file
could not say which readout produced the games. **Provenance that lives only in
a preparation log is provenance the artifact does not carry.**

## 2. THE FIRST REPAIR WAS NOT ENOUGH, AND THE REVIEW SAID SO

The barrier established that **two fresh derivations agree with the frozen
settings**. That is not the claim a record has to support. `frozen_incumbent_
identity()` built one config object; the seam built another inside `play()`,
AFTER the outputs were already open. Same function, same values, **two
objects**, and nothing compared the record with the one actually used.

### ONE OBJECT, ONE ORIGIN

```
run_pilot            constructs it ONCE, like the deadline's one origin
  -> _production_play(..., config)   carries it as `play.config` and USES it;
                                     refuses rather than constructing its own
  -> _run_pilot_unguarded            reads the identity off `play.config` -- the
                                     object the seam will hand the builder --
                                     BEFORE opening any output
```

`frozen_incumbent_identity(config)` and `check_incumbent_identity(identity,
config)` REQUIRE the object and will not make one. A seam declaring no
configuration is refused rather than handed a fresh one: defaulting there is how
the record came to describe an object nothing played with.

Two drifts are reachable, both meaning the record does not describe what plays,
and both refusing **before an output exists and before a game** — the tests
assert the inert seam's call counter is still zero:

| drift | caught by |
|---|---|
| the OBJECT parts company with the qualified path | the cross-check, first |
| the RECORD describes another object | `_same`, by path, against the object |

## 3. 🔴 TWO MISTAKES OF MINE INSIDE THE REPAIR ITSELF

**The behavioural test compared `==`.** The control "the seam builds a config
instead of using the one it was given" is what proves that was not enough: with
the seam building its own, the captured configuration **compared equal field for
field**, and only `is` caught it. The same mistake in miniature — an equal
object is not the object.

**A guard I wrote was unreachable.** I compared the field SETS of
`argmax_config` and the frozen settings; both take their keys from
`frozen_settings()["eval_config"]`, so no input can separate them. Deleted. The
record's own field set is a different question and `_same` answers it by path; a
test covers that instead.

## 4. THE RETIRED CONTROL, AND WHY ITS REPLACEMENT IS STRONGER

| | retired | replacement |
|---|---|---|
| injection | `argmax_cfg = frozen_argmax_config()` → `G3.eval_config()` | `argmax_cfg = play.config` → `frozen_argmax_config()` |
| caught by | a VALUE difference (`selection_mode` reverted) | OBJECT IDENTITY |
| observed reason | `the builder got 'opening_temperature' but the header would record 'argmax'` | `the builder must receive THE object the identity was read off, not an equal one` |

Under the threaded code the retired control could not have failed at all — the
values agree. Its claim is now made, more tightly, against the new contract.

## 5. RESULTS OF THE RUNS PACKAGED HERE

| run | commit | result |
|---|---|---|
| harness at the barrier (`02`) | `20f2c0e` | **607/607 rejected; 0 not caught; 0 indeterminate; 0 stale; 0 orphans; clean baseline over 496 target tests PASS; PROBLEMS 0**, exit 0 |
| suite at the barrier (`03`) | `20f2c0e` | 4,649 passed / 0 failed / 4 skipped, 968.97 s |
| 🔴 **harness, threaded (`04`)** | `0bd587a` | **608/611; 0 not caught; 0 indeterminate; 3 STALE; PROBLEMS 3, EXIT 4** |
| suite, threaded (`05`) | `0bd587a` | 4,653 passed / 0 failed / 4 skipped, 935.32 s |
| harness, final (`06`) | `3a52f0e` | **610/610 rejected; 0 not caught; 0 indeterminate; 0 stale; 0 orphan reasons; 0 duplicate labels; clean baseline over 499 distinct target tests PASS; PROBLEMS 0**, exit 0 |
| suite, final (`07`) | `3a52f0e` | 4,653 passed / 0 failed / 4 skipped / 53 deselected, 923.10 s, exit 0 |
| pre-run verification (`08`) | `3a52f0e` | **53/53 PASS**, exit 0 |

### 🔴 THE EXIT 4 IS THE RESULT OF RECORD FOR `0bd587a`
A clean suite and a passing pre-run check do not override three stale controls.
The threading moved the exact lines three controls inject into, and the
stale-anchor check caught all three — the mechanism doing its job, and the
fourth round in this programme where a headline count was not the verdict. Two
were re-anchored (their declared reasons re-observed and unchanged); one was
retired as superseded, with its reason removed in the same edit so it could not
become an orphan.

## 6. SCOPE, stated with the result

These establish that the incumbent's identity is derived from and verified
against the configuration object the production seam will hand the builder,
before any output is created; that the durable header carries that identity and
`selection_mode`; and that drift in either direction is refused before play —
**on this tree, at `3a52f0e`, with the gate shut throughout.**

They establish NOTHING about H3's question. The pilot has never played a game,
its launch path has never completed a run, it MAY time out by construction, and
it **cannot produce a strength verdict** in any case: its outcomes are
"authorize design work on a full study" or "close H3".

**See `09_correction.md`** — three claims in this phase's commit messages were
false and are corrected there.
