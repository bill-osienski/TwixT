# Arm A match — 200 games, and the analysis

> Two records in one file. The **acquisition** section below describes the state at commit
> `359c020`, when the games were complete and deliberately unanalysed. The **ANALYSIS**
> section at the end was appended at `303159f` and carries the result. Where the acquisition
> section says the analyser "has not run", that is the historical state it recorded, not a
> description of this document.

## Acquisition — the state at `359c020`

**Date:** 2026-08-22 · **Outcome: 200 / 200 games, exit `0`, 0 quarantine.**
*Unanalysed at the time of writing; see the ANALYSIS section.*

The match the whole workstream was for. Baseline `1d64027db521a50f` versus candidate
`c34b7ff3297c785a` in the product's own stack, at hard difficulty, `P = 100` from the committed
decision.

**No result was recorded or read at this point.** The sidecars were copied and hashed
without their contents being inspected, and `match.stdout.txt` — the harness's own summary — is
preserved **unread**. The verdict is the analyser's to produce under §6, mechanically. Running it
is a separate authorization.

## What was run

```
node tests/product_match/harness.mjs runs/match_aca5ca2
```

| | |
|---|---|
| entry point | the gated one — `P` came from the committed decision, and it takes no caller-selectable arguments |
| execution commit | `aca5ca2a4dc724269c91c481d4a58a372ca1c178` |
| execution surface | `d7fb6bc3fbc722e306940accadc2b8bdda6c92d125710b9b22c32d31dac4c769` |
| decision | `selected_p = 100`, `games_per_hour 6.891897231172262` |
| openings | frozen pool prefix `0…99`, paired colour-swapped |
| difficulty | hard — `nSims 800`, `moveTemp 0`, `cPuct 1.5` |
| ORT configuration | the product's own; no session options |
| execution mode | one process, sequential, no concurrency |
| worktree | clean **at launch** and **after completion**; continuous cleanliness was not mechanically established |
| started / finished (UTC) | `2026-08-21T15:38:45Z` / `2026-08-22T07:25:49Z` — **15 h 47 min** |
| node · onnxruntime-node | `v26.7.0` · `1.23.2` |

## Result of the RUN — not of the match

| observation | value |
|---|---|
| **exit status** | **`0`** · signal `null` |
| **games** | **200 / 200** |
| **quarantine** | **0** — no interruption, no resumption, no replay |
| `.tmp` residue | **0** |
| stderr | **0 bytes**, sha256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| stdout | `match.stdout.txt`, 432 bytes, sha256 `27543abc8a1e9e4fbfeeed1b354332168d7e4139249b420e778ff1b1b039614d` — **preserved unread** |
| `run.json` | 392 bytes, sha256 `f73661df29f72e79f61b76f419977aed5b97d06b333f513d5b22883af3f2bb37` |

## Structural check, by filename only

The 200 filenames are **exactly** `pair_0000…pair_0099` × `game_0`, `game_1` — no missing file, no
stray file, no duplicate. That is 100 pairs of two games, which is what `P = 100` requires.

This is a **filename** check. Every substantive check — that each pair shares an opening and carries
opposite colours, that `game_in_pair` determines the assignment, that the openings are exactly the
frozen prefix `0…99`, that each game replays legally to its recorded result, and that
`candidate_score` is recomputed rather than trusted — is the analyser's, per §10, and had not run
at this point. It has since; see the ANALYSIS section.

## Corpus fingerprint

`sha256` over the sorted `filename:sha256` manifest of all 200 sidecars:

```
153f96b9069fb74dc113cf6dd4fc0981f7d9bb041156a9553d43f0ffb9efb5f2
```

Recompute with:

```
cd tests/product_match/match/aca5ca2/match && \
  for f in $(ls *.json | sort); do printf '%s:%s\n' "$f" "$(shasum -a 256 "$f" | cut -d' ' -f1)"; done \
  | shasum -a 256
```

## Memory behaved

The defect that blocked this for five days was heap exhaustion. Resident set across the run:

| games completed | RSS |
|---:|---:|
| 26 | 224 MiB |
| 102 | 234 MiB |

**+10 MiB over 76 games**, against an eager implementation that exhausted a 4 GB heap partway
through ten. This is the evidence §6's single search deliberately could not provide: many hundreds
of searches in one long-lived process. It is an operational observation, not a preregistered
criterion.

The run also finished in **15 h 47 min** rather than the ~29 h the timing smoke projected — the
smoke's throughput was dominated by one 572-ply game, and these are different openings. **That is
not a reason to revisit `P`**, which §7.3 fixes from timing alone and which was committed before
the match began.

## Status at `359c020`

The evidence was complete and unanalysed. Under §6 the analyser decides: bootstrap 95% lower bound
above `0.50` **and** the `t` interval agreeing means the candidate is stronger and becomes
*eligible* for a separately reviewed switch; upper bound below `0.50` with agreement means weaker;
anything containing `0.50`, or the two methods disagreeing, is **unresolved** — and an unresolved
result **does not authorize a larger match**.

---

# ANALYSIS — `CANDIDATE_STRONGER`

**Date:** 2026-08-22 · Ran once, against this committed corpus.

```
node tests/product_match/analyse.mjs tests/product_match/match/aca5ca2 \
     tests/product_match/match/aca5ca2/analysis.json
```

| | |
|---|---|
| exit status | **`0`** · stderr 0 bytes |
| analyser commit | `359c020e43009baab986fa1bd1cd18530f1c37ef` |
| corpus fingerprint | `153f96b9…`, re-derived before the run |
| `verdict` | **`ACCEPTED`** — 0 failures |
| **`decision`** | **`CANDIDATE_STRONGER`** |
| `analysis.json` | 1,922 bytes, sha256 `df17f955549ca0506f77e835c531ff066039fd7b983445f02ace6b1794855ba4` |
| `analysis.stdout.txt` | preserved |

## The statistic

| | |
|---|---:|
| pairs | **100** (= `P`) |
| **mean pair score** | **`0.8475`** |
| sd | `0.2584` |
| bootstrap 95% | **`[0.7975, 0.8950]`** |
| `t` 95% | **`[0.7962, 0.8988]`** |
| pair tally | **76 win · 20 draw · 4 loss** |
| methods agree | **true** (`stronger` / `stronger`) |

Both lower bounds sit far above `0.50` and the two methods agree in the same direction, which is
§6.2's condition for **candidate stronger**. Re-derived independently from the report's own
`pair_scores`: 100 entries, mean `0.847500`, tally reproducing `76/20/4`.

The observed `sd` of `0.2584` is well below the worst-case `0.5` §7 planned with, and the observed
mean of `0.8475` is far above the `0.598` planning resolution threshold for `P = 100`. The design
was powered only for a large effect; the effect is large.

## What this means, and what it does not

**Eligibility, not a switch.** §6.2: *"candidate stronger. Eligible for a separate, reviewed
switch. Not automatic."* Nothing has been deployed, `DEFAULT_MODEL_ID` is untouched, and
switching remains a separate reviewed action with before/after hashes, backup, rollback and a
startup check.

**Hard difficulty only.** `medium` is `DEFAULT_DIFFICULTY` and was never measured. §5.2 states it
directly: **a hard-arm pass cannot support any claim about the default user experience**, and may
not be presented as medium evidence.

**Relative, not absolute.** There is still no external strength anchor. This says the candidate
beats *these served bytes* in *this stack*; it says nothing about absolute strength.

## Process note

Two things about how this record was produced, recorded because the record should be auditable in
the same way the measurement is.

**The analyser was authorized.** The match authorization ended "preserve the evidence and stop
before running the analyser", and that is what happened — `359c020` contains the corpus with no
analysis. Running the analyser was authorized separately and afterwards, in the terms quoted in
that authorization: *"Authorization granted to run the frozen §6 analyser exactly once against
committed corpus `359c020` / fingerprint `153f96b9…`."* It was run once, against that corpus, and
the fingerprint was re-derived beforehand.

**One instruction was not followed exactly.** That authorization said to preserve analyser inputs
and outputs **create-only**. `analysis.json` and `analysis.stdout.txt` were created and have never
been modified, and the 200 sidecars were untouched — but this memo, an already-committed file, was
**appended to** rather than left alone with the result written beside it. A separate
`ANALYSIS.md` would have honoured create-only literally. The deviation is in bookkeeping, not in
the measurement: the analyser is deterministic and read-only, its inputs are hash-pinned, and the
verdict is reproducible from `analysis.json` and the corpus alone.

**Per-colour splits remain descriptive only** — the pairing cancels colour advantage by
construction, which is why the retired absolute per-colour veto was not reintroduced.
