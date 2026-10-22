# Self-play throughput sweep (24x24, 400 sims, 48 games/run)

Machine: Apple M5 Max, 18 CPUs; git 5ddd7437; weights model_iter_0001.safetensors sha1 209cf2d4fd24

Ranking metric: median steady-state games/h over good repeats, i.e. the game rate while every worker is busy (first `Games: n` line with n >= 5 up to the first `Worker done`), which leaves out start-up and the straggler tail (one long game finishing alone). Runs were stopped at the first `Worker done` (status STEADY), so the trainer's batch and plies stats are absent. Games/h assumes the same game-length mix across settings; a larger eval batch changes the search and may change it. Throughput only, and one clean sweep does not prove the Metal hang is gone.

## Stage 1: workers

| workers | good/runs | steady games/h (median, range) | full-run games/h | avg batch | avg plies | statuses |
|---|---|---|---|---|---|---|
| 10 | 3/3 | 863 (776-998) | - | - | - | STEADY |
| 12 | 3/3 | 1038 (1003-1104) | - | - | - | STEADY |
| 14 | 3/3 | 1068 (874-1235) | - | - | - | STEADY |
| 16 | 3/3 | 1136 (1042-1214) | - | - | - | STEADY |
| 18 | 3/3 | 1207 (1142-1251) | - | - | - | STEADY |

## Stage 2: eval batch

| eval_batch | good/runs | steady games/h (median, range) | full-run games/h | avg batch | avg plies | statuses |
|---|---|---|---|---|---|---|
| 14 | 3/3 | 1207 (1142-1251) | - | - | - | STEADY |
| 16 | 3/3 | 1084 (989-1088) | - | - | - | STEADY |
| 24 | 3/3 | 1137 (1056-1195) | - | - | - | STEADY |
| 32 | 3/3 | 880 (858-1199) | - | - | - | STEADY |
| 48 | 3/3 | 910 (906-1005) | - | - | - | STEADY |

## Picks

{
  "workers_best": 18,
  "workers_knee": 18,
  "eval_batch_best": 14,
  "eval_batch_knee": 14,
  "first_hang_eval_batch": null,
  "within": 0.03
}
