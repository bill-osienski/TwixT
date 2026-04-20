# Probe Suite Baselines

Immutable baseline scorings of historical checkpoints against the curated
probe suite. Used by the validation gate's "improvement vs baseline" rule.

## Files

- `iter_0999_fresh_24ch.csv` — baseline probe scoring for the iter-999
  checkpoint (24-channel format, pre-retrain). Generated once via Phase 0.
  **Never regenerated or overwritten** once committed.

## Adding a new baseline

Baselines are added when:
- A new reference checkpoint is promoted to the comparison set (e.g. the
  first post-retrain checkpoint that clears the gate)
- The probe suite is deliberately amended (an ADR-level decision)

Filenames encode **checkpoint identity**, not just iteration number. Example:
`iter_1500_v2_30ch.csv` — iteration 1500, from the v2 (30-channel) run.

Each baseline must be generated with an **explicit `--weights` path**.
"Use latest checkpoint" is not permitted for baseline generation.

## Schema

Each baseline directory contains:
- `<name>.csv` — per-probe row: probe_id, nn_value, mcts_root_value,
  sign_correct_nn, sign_correct_mcts, magnitude_in_band, search_corrected, both_wrong
- `<name>.json` — aggregate: per-tier sign-correct rates, median magnitudes,
  category breakdowns, timestamp, weights path, probe suite revision
