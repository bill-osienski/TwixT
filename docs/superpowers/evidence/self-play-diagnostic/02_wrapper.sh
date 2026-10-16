#!/bin/sh
# SELF-PLAY DIAGNOSTIC OUTER WRAPPER -- a copy of inference-mode-match/02_wrapper.sh; the diff: the paths, the run
# script and the ceiling. The whole run under ONE 1,200 s ceiling: TERM at 1,190 s, KILL 10 s later (GNU timeout
# signals the run's whole process group).
# Run from anywhere; it works in the tree that holds it. The environment and the checkpoint come from the
# main checkout (the parent of the shared git directory), by absolute path; nothing is installed or copied.
# Writes, create-only: run.stdout.txt, run.stderr.txt, outer.json. Exits with the outer status.
set -u
OUT_REL=docs/superpowers/evidence/self-play-diagnostic
W=$(cd "$(dirname "$0")/../../../.." && pwd -P) || exit 4
cd "$W" || exit 4
MAIN=$(cd "$(git rev-parse --path-format=absolute --git-common-dir)/.." && pwd -P) || exit 4
PY="$MAIN/.venv/bin/python"
CKPT="$MAIN/checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors"
for f in progress.jsonl record.json games run.stdout.txt run.stderr.txt outer.json; do
  [ -e "$OUT_REL/$f" ] && { echo "REFUSED: $OUT_REL/$f exists (one-shot)" >&2; exit 1; }
done
set -C                                   # create-only redirections from here on
start=$(date +%s)
timeout -k 10 1190 env PYTHONPATH=. "$PY" -B "$OUT_REL/01_selfplay.py" --checkpoint "$CKPT" \
  > "$OUT_REL/run.stdout.txt" 2> "$OUT_REL/run.stderr.txt"
status=$?
end=$(date +%s)
survivors=$(pgrep -f "$OUT_REL/01_selfplay.py" | tr '\n' ' ')
printf '{"outer_exit": %d, "wall_seconds": %d, "survivors": "%s", "tree": "%s", "python": "%s", "checkpoint": "%s"}\n' \
  "$status" "$((end - start))" "$survivors" "$W" "$PY" "$CKPT" > "$OUT_REL/outer.json"
exit "$status"
