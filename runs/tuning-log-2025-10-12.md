# Heuristic tuning log (2025-10-12)

## Adjustment A1 — 2025-10-13T00:52Z

- Changes applied to `assets/js/ai/search.json`:
  - `redGlobalMultiplier`: 1.18 → 1.16
  - `redBaseBonus`: 150 → 142
  - `redSpanGainMultiplier`: 2.3 → 2.34
  - `redFinishPenaltyFactor`: 0.78 → 0.75
  - `blackFinishScaleMultiplier`: 0.96 → 0.94
  - `blackSpanGainMultiplier`: 1.02 → 0.99
  - `blackDoubleCoverageScale`: 0.80 → 0.77
- Depth-2 sample (36 games, runId `1760316720598`): red 22 — black 13 — draws 1.
- Depth-3 sample (36 games, runId `1760316778277`): red 15 — black 20 — draws 1.
- Takeaways: depth 2 still red-leaning; depth 3 still black-leaning but gap narrowed. Plan further tweaks to damp early red base bias while trimming black edge rewards a bit more.
