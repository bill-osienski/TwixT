# Self-play throughput sweep (24x24, 400 sims, 36 games/run)

Machine: Apple M5 Max, 18 CPUs; git 6d888aa8; weights model_iter_0001.safetensors sha1 209cf2d4fd24

Ranking metric: median self-play plies/s (400-sim searches per second, all workers) over OK repeats; positions/s counts the trainer's stored positions, which are not one per ply. Throughput only: a larger eval batch changes the search, and one clean sweep does not prove the Metal hang is gone.

## Stage 1: workers

| workers | OK/runs | plies/s (median, range) | pos/s | games/h | s/game | avg batch | avg plies | statuses |
|---|---|---|---|---|---|---|---|---|
| 10 | 1/2 | 14.80 (14.80-14.80) | 22.3 | 952 | 3.8 | 11.5 | 56.0 | OK, TIMEOUT |
| 12 | 2/2 | 14.41 (14.12-14.70) | 21.7 | 937 | 3.8 | 11.6 | 55.4 | OK |
| 14 | 1/2 | 13.48 (13.48-13.48) | 20.2 | 784 | 4.6 | 11.6 | 61.9 | OK, TIMEOUT |
| 16 | 1/2 | 13.87 (13.87-13.87) | 20.7 | 924 | 3.9 | 11.6 | 54.0 | OK, TIMEOUT |
| 18 | 1/2 | 5.84 (5.84-5.84) | 8.7 | 348 | 10.3 | 11.6 | 60.4 | OK, TIMEOUT |

## Stage 2: eval batch

| eval_batch | OK/runs | plies/s (median, range) | pos/s | games/h | s/game | avg batch | avg plies | statuses |
|---|---|---|---|---|---|---|---|---|
| 14 | 1/2 | 14.80 (14.80-14.80) | 22.3 | 952 | 3.8 | 11.5 | 56.0 | OK, TIMEOUT |
| 16 | 1/2 | 6.85 (6.85-6.85) | 10.2 | 370 | 9.7 | 12.5 | 66.6 | OK, TIMEOUT |
| 24 | 0/1 | - | - | - | - | - | - | SKIPPED_AFTER_HANG_AT_16 |
| 32 | 0/1 | - | - | - | - | - | - | SKIPPED_AFTER_HANG_AT_16 |
| 48 | 0/1 | - | - | - | - | - | - | SKIPPED_AFTER_HANG_AT_16 |

## Picks

{
  "workers_best": 10,
  "workers_knee": 10,
  "eval_batch_best": 14,
  "eval_batch_knee": 14,
  "first_hang_eval_batch": 16,
  "within": 0.03
}
