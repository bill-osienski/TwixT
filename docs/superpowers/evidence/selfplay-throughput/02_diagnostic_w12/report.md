# Self-play throughput sweep (24x24, 400 sims, 36 games/run)

Machine: Apple M5 Max, 18 CPUs; git 6864a16e; weights model_iter_0001.safetensors sha1 209cf2d4fd24

Ranking metric: median self-play plies/s (400-sim searches per second, all workers) over OK repeats; positions/s counts the trainer's stored positions, which are not one per ply. Throughput only: a larger eval batch changes the search, and one clean sweep does not prove the Metal hang is gone.

## Stage 1: workers

| workers | OK/runs | plies/s (median, range) | pos/s | games/h | s/game | avg batch | avg plies | statuses |
|---|---|---|---|---|---|---|---|---|
| 12 | 6/6 | 8.14 (4.76-14.86) | 12.2 | 474 | 8.4 | 11.6 | 62.0 | OK |

## Stage 2: eval batch

| eval_batch | OK/runs | plies/s (median, range) | pos/s | games/h | s/game | avg batch | avg plies | statuses |
|---|---|---|---|---|---|---|---|---|

## Picks

{
  "workers_best": 12,
  "workers_knee": 12,
  "eval_batch_best": null,
  "eval_batch_knee": null,
  "first_hang_eval_batch": null,
  "within": 0.03
}
