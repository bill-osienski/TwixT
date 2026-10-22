# Self-play throughput sweep (24x24, 400 sims, 48 games/run)

Machine: Apple M5 Max, 18 CPUs; git 5ddd7437; weights model_iter_0001.safetensors sha1 209cf2d4fd24

Ranking metric: median steady-state games/h over good repeats, i.e. the game rate while every worker is busy (first `Games: n` line with n >= 5 up to the first `Worker done`), which leaves out start-up and the straggler tail (one long game finishing alone). Runs went to the end (--full). Games/h assumes the same game-length mix across settings; a larger eval batch changes the search and may change it. Throughput only, and one clean sweep does not prove the Metal hang is gone.

## Stage 1: workers

| workers | good/runs | steady games/h (median, range) | full-run games/h | avg batch | avg plies | statuses |
|---|---|---|---|---|---|---|
| 18 | 1/1 | 1219 (1219-1219) | 910 | 10.5 | 55.5 | OK |

## Stage 2: eval batch

| eval_batch | good/runs | steady games/h (median, range) | full-run games/h | avg batch | avg plies | statuses |
|---|---|---|---|---|---|---|

## Picks

{
  "workers_best": 18,
  "workers_knee": 18,
  "eval_batch_best": null,
  "eval_batch_knee": null,
  "first_hang_eval_batch": null,
  "within": 0.03
}
