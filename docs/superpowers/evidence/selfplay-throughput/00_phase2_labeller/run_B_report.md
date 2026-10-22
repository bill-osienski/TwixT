# Phase 2 Knob Sweep — Coordinate Descent

Baseline wallclock: 336.92s

## Coordinate descent trace

| Pass | Knob       | Config                       | Wallclock          | Speedup | Status |
|------|------------|------------------------------|--------------------|---------|--------|
|    - | baseline   | serial w=1 eb=14 sf=16       | 336.92s            | 1.00x   | OK |
|    1 | workers    | process w=12 eb=14 sf=16     | 102.83s            | 3.28x   | OK *winner* |
|    1 | eval_batch | process w=12 eb=14 sf=16     | (cached: 102.83s)  | 3.28x   | OK *winner* |
|    1 | eval_batch | process w=12 eb=16 sf=16 *unsafe* | 102.99s            | 3.27x   | OK |
|    1 | eval_batch | process w=12 eb=24 sf=16 *unsafe* | 103.36s            | 3.26x   | OK |
|    1 | eval_batch | process w=12 eb=32 sf=16 *unsafe* | 103.30s            | 3.26x   | OK |
|    1 | eval_batch | process w=12 eb=48 sf=16 *unsafe* | 103.49s            | 3.26x   | OK |
|    1 | stall_flush | process w=12 eb=14 sf=16     | (cached: 102.83s)  | 3.28x   | OK *winner* |

Last pass executed: 1.

## Recommendation

Best config: `--label-worker-mode process --label-workers 12 --mcts-eval-batch-size 14 --mcts-stall-flush-sims 16`
Wallclock: 102.83s (3.28x speedup vs. 336.92s serial baseline)
Admitted IDs: same as baseline (yes)
Borderline reruns: 2
Borderline flips: 0