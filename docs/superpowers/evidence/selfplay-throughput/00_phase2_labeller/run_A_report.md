# Phase 2 Knob Sweep — Coordinate Descent

Baseline wallclock: 336.76s

## Coordinate descent trace

| Pass | Knob       | Config                       | Wallclock          | Speedup | Status |
|------|------------|------------------------------|--------------------|---------|--------|
|    - | baseline   | serial w=1 eb=14 sf=16       | 336.76s            | 1.00x   | OK |
|    1 | workers    | process w=8 eb=14 sf=16      | 107.63s            | 3.13x   | OK |
|    1 | workers    | process w=10 eb=14 sf=16     | 103.71s            | 3.25x   | OK |
|    1 | workers    | process w=12 eb=14 sf=16     | 103.17s            | 3.26x   | OK |
|    1 | workers    | process w=14 eb=14 sf=16     | 103.80s            | 3.24x   | OK |
|    1 | workers    | process w=16 eb=14 sf=16     | 103.77s            | 3.25x   | OK |
|    1 | workers    | process w=18 eb=14 sf=16     | 102.94s            | 3.27x   | OK *winner* |
|    1 | eval_batch | process w=18 eb=12 sf=16     | 103.19s            | 3.26x   | OK |
|    1 | eval_batch | process w=18 eb=14 sf=16     | (cached: 102.94s)  | 3.27x   | OK *winner* |
|    1 | stall_flush | process w=18 eb=14 sf=8      | 103.15s            | 3.26x   | OK |
|    1 | stall_flush | process w=18 eb=14 sf=16     | (cached: 102.94s)  | 3.27x   | OK *winner* |
|    1 | stall_flush | process w=18 eb=14 sf=32     | 103.33s            | 3.26x   | OK |
|    2 | workers    | process w=8 eb=14 sf=16      | (cached: 107.63s)  | 3.13x   | OK |
|    2 | workers    | process w=10 eb=14 sf=16     | (cached: 103.71s)  | 3.25x   | OK |
|    2 | workers    | process w=12 eb=14 sf=16     | (cached: 103.17s)  | 3.26x   | OK |
|    2 | workers    | process w=14 eb=14 sf=16     | (cached: 103.80s)  | 3.24x   | OK |
|    2 | workers    | process w=16 eb=14 sf=16     | (cached: 103.77s)  | 3.25x   | OK |
|    2 | workers    | process w=18 eb=14 sf=16     | (cached: 102.94s)  | 3.27x   | OK *winner* |
|    2 | eval_batch | process w=18 eb=12 sf=16     | (cached: 103.19s)  | 3.26x   | OK |
|    2 | eval_batch | process w=18 eb=14 sf=16     | (cached: 102.94s)  | 3.27x   | OK *winner* |
|    2 | stall_flush | process w=18 eb=14 sf=8      | (cached: 103.15s)  | 3.26x   | OK |
|    2 | stall_flush | process w=18 eb=14 sf=16     | (cached: 102.94s)  | 3.27x   | OK *winner* |
|    2 | stall_flush | process w=18 eb=14 sf=32     | (cached: 103.33s)  | 3.26x   | OK |

Last pass executed: 2.

## Recommendation

Best config: `--label-worker-mode process --label-workers 18 --mcts-eval-batch-size 14 --mcts-stall-flush-sims 16`
Wallclock: 102.94s (3.27x speedup vs. 336.76s serial baseline)
Admitted IDs: same as baseline (yes)
Borderline reruns: 2
Borderline flips: 0