# Correlation-Driven Tuning TODO

## Core Implementation
1. **Data extraction + weighting** ✅ _implemented via `build_model_samples` and helpers_
   - Parse last 20 cycles of combo + validation logs.
   - For each hash/depth sample compute wins, draws, games, and age = current_cycle − sample_cycle.
   - Weight = games × phase factor (1.0 validation, 0.2 combo) × exp(−age/5).

2. **Feature prep + ridge models** ✅ _ridge solver + CLI (`autoTune correlate`)_
   - Z-score knobs over that window (optionally weighted).
   - Fit weighted ridge models (depth2 & depth3, λ=0.5) on bias = (wins_red − wins_black)/(wins_red + wins_black), draws excluded.
   - Persist coefficients, intercept, normalization stats, R², and effective sample counts per depth.

3. **Gradient-guided bucket seeding** ✅ _`command_suggest` now loads correlation models, re-ranks buckets, and injects model-guided variants_
   - Use per-depth gradients to steer `best`/`soft-best`/`trend` configs, re-ranking by predicted parity.
   - Center mutate/trend probes near correlation-favored regions while keeping ~30% exploration (randoms, multi-knob mutates, single-knob probes).

4. **Auto-drop + review gating** ✅ _hash performance summary + policy enforcement in `command_suggest`_
   - Compute weighted deltas vs baseline (current best/reference config) for each hash/depth.
   - Auto-drop when effective games ≥500 and delta ≤ −0.03 with CI excluding 0.
   - Soft-flag when effective games ≥300 and delta ≤ −0.02 for de-prioritization/review.

5. **Diagnostics + reporting** ✅ _`correlation-state.json` and `command_report` now surface model/flag summaries_
   - Emit per-cycle reports: coefficient trends, predicted vs actual parity by depth, hashes flagged/dropped with stats.

## Risk Mitigations
- Offline backtest on historical cycles before live use.
- Shadow mode logging for several cycles prior to enabling control.
- Unit/regression tests with synthetic data.
- Safety clamps on per-cycle knob movement and keep one baseline validation config.
- Manual override hooks with diagnostic transparency.
