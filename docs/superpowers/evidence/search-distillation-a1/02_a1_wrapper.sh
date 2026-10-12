#!/bin/sh
# A1 OUTER WRAPPER -- the whole run under ONE 1,800 s ceiling: TERM at 1,790 s, KILL 10 s later (GNU
# timeout signals the run's whole process group). Card: docs/superpowers/2026-10-11-search-distillation-a1-card.md.
# Run from anywhere; it works in the tree that holds it. The environment and the checkpoint come from the
# main checkout (the parent of the shared git directory), by absolute path; nothing is installed or copied.
# Writes, create-only: run.stdout.txt, run.stderr.txt, outer.json. Exits with the outer status.
set -u
OUT_REL=docs/superpowers/evidence/search-distillation-a1
W=$(cd "$(dirname "$0")/../../../.." && pwd -P) || exit 4
cd "$W" || exit 4
MAIN=$(cd "$(git rev-parse --path-format=absolute --git-common-dir)/.." && pwd -P) || exit 4
PY="$MAIN/.venv/bin/python"
CKPT="$MAIN/checkpoints/alphazero-v2-calib020-from0409/model_iter_0001.safetensors"
for f in progress.jsonl record.json run.stdout.txt run.stderr.txt outer.json; do
  [ -e "$OUT_REL/$f" ] && { echo "REFUSED: $OUT_REL/$f exists (one-shot)" >&2; exit 1; }
done
set -C                                   # create-only redirections from here on
start=$(date +%s)
timeout -k 10 1790 env PYTHONPATH=. "$PY" -B "$OUT_REL/01_a1_run.py" --checkpoint "$CKPT" \
  > "$OUT_REL/run.stdout.txt" 2> "$OUT_REL/run.stderr.txt"
status=$?
end=$(date +%s)
survivors=$(pgrep -f "$OUT_REL/01_a1_run.py" | tr '\n' ' ')
printf '{"outer_exit": %d, "wall_seconds": %d, "survivors": "%s", "tree": "%s", "python": "%s", "checkpoint": "%s"}\n' \
  "$status" "$((end - start))" "$survivors" "$W" "$PY" "$CKPT" > "$OUT_REL/outer.json"
exit "$status"
