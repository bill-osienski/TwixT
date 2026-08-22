# Arm A match — 200 games COMPLETE, unanalysed

**Date:** 2026-08-22 · **Outcome: 200 / 200 games, exit `0`, 0 quarantine. NOT YET ANALYSED.**

The match the whole workstream was for. Baseline `1d64027db521a50f` versus candidate
`c34b7ff3297c785a` in the product's own stack, at hard difficulty, `P = 100` from the committed
decision.

**No result is recorded here, and none has been read.** The sidecars were copied and hashed
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
| worktree | clean at launch and throughout |
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
`candidate_score` is recomputed rather than trusted — is the analyser's, per §10, and has not run.

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

## Status

The evidence is complete and unanalysed. Under §6 the analyser decides: bootstrap 95% lower bound
above `0.50` **and** the `t` interval agreeing means the candidate is stronger and becomes
*eligible* for a separately reviewed switch; upper bound below `0.50` with agreement means weaker;
anything containing `0.50`, or the two methods disagreeing, is **unresolved** — and an unresolved
result **does not authorize a larger match**.
