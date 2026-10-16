# TwixT_Game — standing instructions

## Experiment ledger (keep it current)

`docs/alphazero-value-search-experiment-ledger.md` is the append-only record of every experiment. Whenever an
experiment, screen, match, diagnostic or probe is **run and judged, or closed**, update the ledger in the same
series of commits — before reporting the line as done:

1. **Long-form entry** under the latest "Post-closure work" section (or a new dated section), with:
   test · hypothesis · why it was plausible · prediction on record · what ran (commit hashes) · result ·
   did it match the prediction, and why · conclusion / status.
   If no prediction was stated before the run, write "None stated in advance" — never backfill one.
2. **Top banner** (the `>` blockquotes near the top): one line per closed or open line, key conclusion current.
3. **Do-not-repeat** list: a new numbered item for every closed line — what must not be retried, extended or
   rescued, and why.

Never rewrite or delete existing ledger text; corrections go in as new, dated text.

## Inference mode

Before `c8cb2aa6`, `eval_runner`'s default factory ran BatchNorm in training mode, and `probe_eval.py`'s entry
point still does (ledger, Post-closure entry 3). Do not reuse a pre-fix number as a baseline for an eval-mode run;
re-measure the baseline.
